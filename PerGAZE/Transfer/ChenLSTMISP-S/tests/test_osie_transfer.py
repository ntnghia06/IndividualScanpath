import copy
from contextlib import contextmanager
import shutil
import uuid
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch import nn

BRANCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BRANCH / "src"))
import bootstrap
from osie_data import read_records, split_test, record_identity, SubjectAdaptationDataset
from adaptation import build_manifest, expand_subject_embeddings, freeze_for_adaptation, require_embedding_checkpoint
import fewshot
from dataset.tasks import HEAD_SCHEMA, OBJECT_TO_INDEX


@contextmanager
def fixture_directory():
    # Plain mkdir avoids restrictive Windows mkdtemp ACLs in sandboxed runs.
    parent = (BRANCH / "tests").resolve()
    root = parent / ("work_" + uuid.uuid4().hex)
    root.mkdir()
    try:
        yield root
    finally:
        if root.resolve().parent != parent or not root.name.startswith("work_"):
            raise RuntimeError("Refusing cleanup outside the test workspace")
        shutil.rmtree(root)


def records(subjects=2, images=4):
    return [{"name": f"{1000+i}.jpg", "subject": subject, "split": "test",
             "X": [400., 799.], "Y": [300., 599.], "T": [200., 250.], "length": 2}
            for subject in range(1, subjects + 1) for i in range(images)]


class TinyNetwork(nn.Module):
    """Replace only architecture size; exercise real loss, metrics and checkpoint lifecycle."""
    def __init__(self, config, manifest):
        super().__init__()
        self.subject_num = len(manifest["subjects"])
        self.subject_embedding = nn.Embedding(self.subject_num, config["embedding_dim"])
        self.backbone = nn.Sequential(nn.Conv2d(3, 4, 1), nn.BatchNorm2d(4), nn.ReLU())
        self.head = nn.Linear(8, 1201)
        self.duration = nn.Linear(8, 2)
        self.steps = config["max_length"]

    def forward(self, images, subjects, attention_maps, task_heads):
        visual = self.backbone(images[:, :, ::24, ::32]).mean((2, 3))
        fused = torch.cat([visual, self.subject_embedding(subjects)], dim=-1)
        actions = self.head(fused)[:, None, :].expand(-1, self.steps, -1)
        durations = self.duration(fused)[:, None, :].expand(-1, self.steps, -1)
        return {"actions": actions, "all_actions_prob": actions.softmax(-1),
                "log_normal_mu": durations[..., 0] * .01 - 1.5,
                "log_normal_sigma2": durations[..., 1].sigmoid() * .01 + .001}


class OSIEEmbeddingTransferTests(unittest.TestCase):
    def test_real_split_counts_reproducibility_and_no_overlap(self):
        rows = read_records(bootstrap.ROOT.parents[2] / "Experiments/osie/osie_fixations_test.json")
        support, remaining, selected = split_test(rows, 5, 10)
        self.assertEqual((len(support), len(remaining), len(selected)), (75, 975, 15))
        self.assertEqual(selected, split_test(list(reversed(rows)), 5, 10)[2])
        self.assertNotEqual(selected, split_test(rows, 5, 11)[2])
        self.assertFalse({record_identity(r) for r in support} & {record_identity(r) for r in remaining})
        self.assertEqual({record_identity(r) for r in support + remaining}, {record_identity(r) for r in rows})
        with self.assertRaises(ValueError):
            split_test(rows, 70, 10)
        with self.assertRaises(ValueError):
            split_test(rows + [rows[0]], 5, 10)

    def test_targets_and_new_subject_initialization(self):
        with fixture_directory() as folder:
            rows = records(1, 3)
            Image.new("RGB", (800, 600), (80, 120, 60)).save(Path(folder) / rows[0]["name"])
            old = {"ta:1": 0, "air:1": 1}
            manifest = build_manifest({"subjects": old}, rows[:1], rows[1:])
            item = SubjectAdaptationDataset(rows, manifest, folder, max_length=3)[0]
            self.assertEqual(tuple(item["images"].shape), (3, 240, 320))
            self.assertEqual(item["subjects"].item(), 2)
            self.assertEqual(item["task_heads"].item(), -1)
            self.assertEqual(item["attention_maps"].count_nonzero(), 0)
            np.testing.assert_allclose(item["fix_vectors"][0], [160, 120, .2])
            self.assertEqual(item["target_scanpaths"][0, 1 + 15 * 40 + 20], 1)
            self.assertEqual(item["target_scanpaths"][2, 0], 1)
            np.testing.assert_array_equal(item["duration_masks"], [1, 1, 0])
            net = TinyNetwork({"embedding_dim": 4, "max_length": 2}, {"subjects": old})
            initial = net.subject_embedding.weight.detach().clone()
            expand_subject_embeddings(net, old, manifest["subjects"])
            torch.testing.assert_close(net.subject_embedding.weight[:2], initial)
            torch.testing.assert_close(net.subject_embedding.weight[2], initial.mean(0))
            net.requires_grad_(False).eval()
            indices, hook = freeze_for_adaptation(net, manifest)
            self.assertEqual(indices, [2])
            self.assertEqual([name for name, p in net.named_parameters() if p.requires_grad],
                             ["subject_embedding.weight"])
            self.assertFalse(net.training)
            self.assertFalse(net.backbone[1].training)
            net.subject_embedding.weight.sum().backward()
            torch.testing.assert_close(net.subject_embedding.weight.grad[:2], torch.zeros_like(initial))
            torch.testing.assert_close(net.subject_embedding.weight.grad[2], torch.ones(4))
            hook.remove()
            with self.assertRaises(ValueError):
                require_embedding_checkpoint({"adaptation_mode": "osie_full_model_v1"})

    def test_train_resume_reevaluate_and_reject_changed_head(self):
        with fixture_directory() as folder:
            root = Path(folder)
            image_dir = root / "images"
            image_dir.mkdir()
            rows = records(2, 3)
            for name in {r["name"] for r in rows}:
                pixels = np.random.default_rng(int(Path(name).stem)).integers(0, 256, (600, 800, 3), dtype=np.uint8)
                Image.fromarray(pixels).save(image_dir / name)
            test_file = root / "test.json"
            test_file.write_text(json.dumps(rows))
            config = {"model": "chenlstm", "max_length": 2, "min_length": 1,
                      "embedding_dim": 4, "action_map_num": 4, "dropout": .2,
                      "blur_sigma": None, "lambda_1": 1.}
            manifest = {"subjects": {"ta:1": 0, "air:1": 1}, "task_head_schema": HEAD_SCHEMA,
                        "object_to_head": OBJECT_TO_INDEX.copy()}
            initial = TinyNetwork(config, manifest)
            checkpoint = root / "base.pth"
            torch.save({"config": config, "manifest": manifest, "model": initial.state_dict()}, checkpoint)
            args = SimpleNamespace(checkpoint=checkpoint, test_file=test_file, img_dir=image_dir,
                att_dir=None, k=1, seed=10, eval_seed=10, epoch=1, lr=1e-3, batch=2,
                test_batch=2, workers=0, weight_decay=0., clip=12.5, init="mean", task_head=-1,
                device="cpu", gpu_ids=None, eval_repeat_num=1, greedy=True, max_batches=0,
                eval_max_batches=0, log_root=root / "run", resume=False, prepare_only=False)
            with patch.object(fewshot, "parse_args", side_effect=lambda: copy.deepcopy(args)), \
                 patch.object(fewshot, "make_network", side_effect=TinyNetwork):
                fewshot.main()
                last = torch.load(args.log_root / "checkpoints/checkpoint.pth", weights_only=True)
                for name, value in initial.state_dict().items():
                    if name != "subject_embedding.weight":
                        torch.testing.assert_close(last["model"][name], value, rtol=0, atol=0)
                torch.testing.assert_close(last["model"]["subject_embedding.weight"][:2],
                                           initial.subject_embedding.weight, rtol=0, atol=0)
                mean = initial.subject_embedding.weight.detach().mean(0)
                self.assertFalse(torch.equal(last["model"]["subject_embedding.weight"][2], mean))
                self.assertEqual(last["model"]["backbone.1.num_batches_tracked"].item(), 0)
                self.assertEqual(last["manifest"]["adaptation_mode"], "osie_subject_embedding_v1")
                self.assertEqual(last["report"]["trainable_elements"], 8)
                self.assertEqual((last["report"]["support_samples"], last["report"]["test_samples"]), (2, 4))
                self.assertEqual(set(last["report"]["best_metrics"]["by_condition"]), {"freeview"})
                args.resume, args.epoch = True, 2
                fewshot.main()
                last = torch.load(args.log_root / "checkpoints/checkpoint.pth", weights_only=True)
                self.assertEqual(last["epoch"], 1)
                self.assertEqual(len(last["report"]["history"]), 2)
                for name, value in initial.state_dict().items():
                    if name != "subject_embedding.weight":
                        torch.testing.assert_close(last["model"][name], value, rtol=0, atol=0)
                torch.testing.assert_close(last["model"]["subject_embedding.weight"][:2],
                                           initial.subject_embedding.weight, rtol=0, atol=0)
                args.task_head = 0
                with self.assertRaises(ValueError):
                    fewshot.main()
                args.task_head = -1
            spec = importlib.util.spec_from_file_location("osie_eval_cli", BRANCH / "src/test.py")
            cli = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cli)
            args.checkpoint = args.log_root / "checkpoints/best.pth"
            with patch.object(cli, "parse_args", return_value=copy.deepcopy(args)), \
                 patch.object(cli, "make_network", side_effect=TinyNetwork):
                cli.main()
            report = json.loads((args.log_root / "report.json").read_text())
            self.assertEqual(report["reevaluation"]["test_samples"], 4)
            support = {record_identity(r) for r in json.loads((args.log_root / "support.json").read_text())}
            predictions = report["reevaluation"]["predictions"]
            self.assertEqual(len(predictions), 4)
            self.assertFalse(support & {record_identity(p) for p in predictions})
            self.assertEqual(predictions[0]["original_size"], [800, 600])


if __name__ == "__main__":
    unittest.main()
