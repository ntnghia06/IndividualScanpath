import copy
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

import numpy as np
from PIL import Image
import torch
from torch import nn

from gazeformer_transfer import engine
from gazeformer_transfer.adaptation import build_manifest, configure, expand_embeddings, require_mode
from gazeformer_transfer.data import COCOFVDataset, read_records, split_test, record_identity
from gazeformer_transfer.options import parse_args
from gazeformer_transfer import text
from geometry import GEOMETRY, ACTION_COUNT

TRANSFER = Path(__file__).resolve().parents[2]


@contextmanager
def fixture_directory():
    parent = Path(__file__).resolve().parent
    root = parent / ("work_" + uuid.uuid4().hex)
    root.mkdir()
    try:
        yield root
    finally:
        if root.resolve().parent != parent or not root.name.startswith("work_"):
            raise RuntimeError("Refusing cleanup outside the test workspace")
        shutil.rmtree(root)


def rows():
    return [{"name": f"{1000+i}.jpg", "subject": s, "X": [400., 799.],
             "Y": [300., 599.], "T": [200., 250.], "length": 2, "split": "test", "condition": "freeview"}
            for s in (101, 102) for i in range(3)]


def archive(path, encoder=text.ENCODER, prompt=text.PROMPT):
    np.savez(path, tasks=np.asarray([prompt]), vectors=np.ones((1, 768), dtype=np.float32),
             encoder=np.asarray(encoder))


class TinyNetwork(nn.Module):
    def __init__(self, config, manifest):
        super().__init__()
        self.subject_embed = nn.Embedding(len(manifest["subjects"]), config["embedding_dim"])
        self.backbone = nn.Sequential(nn.Conv2d(3, 4, 1), nn.BatchNorm2d(4), nn.ReLU())
        self.text_layer = nn.Linear(768, 4)
        self.head = nn.Linear(12, ACTION_COUNT)
        self.duration = nn.Linear(12, 2)
        self.steps = config["max_length"]

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, images, subjects, attention_maps, task_tokens, task_mask, task_embeddings=None):
        with torch.no_grad():
            visual = self.backbone(images[:, :, ::48, ::64]).mean((2, 3))
        features = torch.cat([visual, self.subject_embed(subjects), self.text_layer(task_embeddings)], -1)
        actions = self.head(features)[:, None, :].expand(-1, self.steps, -1)
        durations = self.duration(features)[:, None, :].expand(-1, self.steps, -1)
        return {"actions" if self.training else "all_actions_prob": actions if self.training else actions.softmax(-1),
            "log_normal_mu": durations[..., 0] * .01 - 1.5,
            "log_normal_sigma2": durations[..., 1].sigmoid() * .01 + .001}


class TransferTests(unittest.TestCase):
    def test_actual_split_matches_chenlstm_and_is_disjoint(self):
        path = TRANSFER.parents[4] / "Experiments/coco_freeview/COCOFreeView_fixations_test.json"
        data = read_records(path)
        support, remaining, selection = split_test(data, 5, 10)
        self.assertEqual((len(support), len(remaining), len(selection)), (50, 6412, 10))
        self.assertEqual(selection, split_test(list(reversed(data)), 5, 10)[2])
        self.assertFalse({record_identity(r) for r in support} & {record_identity(r) for r in remaining})
        spec = importlib.util.spec_from_file_location("chen_transfer_data", TRANSFER / "ChenLSTMISP-FT/src/cocofv_data.py")
        chen = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(chen)
        self.assertEqual(selection, chen.split_test(data, 5, 10)[2])

    def test_text_preparation_reuses_archive_and_rejects_wrong_metadata(self):
        with fixture_directory() as root:
            output = root / "text.npz"
            encoder = unittest.mock.Mock()
            encoder.eval.return_value = encoder
            encoder.encode.return_value = np.ones((1, 768), dtype=np.float32)
            constructor = unittest.mock.Mock(return_value=encoder)
            with patch.dict(sys.modules, {"sentence_transformers": SimpleNamespace(SentenceTransformer=constructor)}):
                text.main(["--output", str(output)])
                text.main(["--output", str(output)])
            constructor.assert_called_once_with(text.ENCODER, device="cpu")
            self.assertEqual(text.load_vector(output).shape, (768,))
            archive(output, encoder="wrong")
            with self.assertRaises(ValueError):
                text.load_vector(output)
            archive(output, prompt="different instruction")
            with self.assertRaises(ValueError):
                text.load_vector(output)

    def test_geometry_freeview_text_and_stop_targets(self):
        with fixture_directory() as root:
            data = rows()
            Image.new("RGB", (800, 600), (80, 120, 60)).save(root / data[0]["name"])
            manifest = build_manifest({"subjects": {"ta:1": 0}}, data[:1], data[1:3], "embedding")
            vector = np.arange(768, dtype=np.float32)
            item = COCOFVDataset(data[:3], manifest, root, vector, max_length=3)[0]
            self.assertEqual(tuple(item["images"].shape), (3, 768, 1024))
            self.assertEqual(item["attention_maps"].shape, (1, 24, 32))
            self.assertNotIn("task_heads", item)
            torch.testing.assert_close(item["task_embeddings"], torch.from_numpy(vector))
            np.testing.assert_allclose(item["fix_vectors"][0], [256, 176, .2])
            self.assertEqual(item["target_scanpaths"].shape, (3, 769))
            self.assertEqual(item["target_scanpaths"][0, 1 + 12 * 32 + 16], 1)
            self.assertEqual(item["target_scanpaths"][2, 0], 1)
            self.assertEqual(item["metadata"]["subject_key"], "cocofv:101")

    def test_real_gazeformer_eval_forward_retains_embedding_gradients(self):
        config = {"model": "gazeformer", "text_dim": 768, "text_embeddings": "text.npz",
            "hidden_dim": 16, "embedding_dim": 4, "max_length": 2, "min_length": 1,
            "nhead": 4, "num_encoder": 1, "num_decoder": 1, "encoder_dropout": 0.,
            "decoder_dropout": 0., "dropout": 0., "action_map_num": 2}
        old = {"ta:1": 0}
        network = engine.make_network(config, {"subjects": old}, {"backbone.marker": torch.zeros(1)})
        # Use a small image feature extractor; the transformer, heads and forward are original.
        network.backbone = nn.Sequential(nn.Conv2d(3, 2048, 1), nn.BatchNorm2d(2048))
        manifest = build_manifest({"subjects": old}, rows()[:1], rows()[1:3], "embedding")
        expand_embeddings(network, old, manifest["subjects"], "mean")
        parameters, _, hook = configure(network, manifest, "embedding")
        prediction = network(torch.rand(1, 3, 24, 32), torch.tensor([1]), torch.zeros(1, 1, 24, 32),
                             torch.zeros(1, 8, dtype=torch.long), torch.ones(1), torch.ones(1, 768))
        self.assertEqual(prediction["all_actions_prob"].shape, (1, 2, 769))
        loss = -prediction["all_actions_prob"][0, 0, 10].clamp_min(1e-12).log()
        loss.backward()
        self.assertGreater(network.subject_embed.weight.grad[1].abs().sum().item(), 0)
        self.assertEqual(network.subject_embed.weight.grad[0].count_nonzero(), 0)
        self.assertTrue(all(p.grad is None for n, p in network.named_parameters() if n != "subject_embed.weight"))
        self.assertEqual(len(parameters), 1)
        hook.remove()

    def test_training_resume_and_reevaluation_for_both_modes(self):
        for mode, branch_name in (("predictor", "GazeformerISP-FT"), ("embedding", "GazeformerISP-S")):
            with self.subTest(mode=mode), fixture_directory() as root:
                torch.manual_seed(2)
                images = root / "images"
                images.mkdir()
                data = rows()
                for name in {r["name"] for r in data}:
                    pixels = np.random.default_rng(int(Path(name).stem)).integers(0, 256, (600, 800, 3), dtype=np.uint8)
                    Image.fromarray(pixels).save(images / name)
                source = root / "test.json"
                source.write_text(json.dumps(data))
                text_file = root / "text.npz"
                archive(text_file)
                config = {"model": "gazeformer", "max_length": 2, "min_length": 1,
                    "embedding_dim": 4, "blur_sigma": None, "lambda_1": 1.,
                    "text_dim": 768, "text_embeddings": "source.npz", "max_text_length": 8}
                manifest = {"subjects": {"ta:1": 0, "air:1": 1}, "geometry": GEOMETRY.copy()}
                initial = TinyNetwork(config, manifest)
                checkpoint = root / "base.pth"
                torch.save({"config": config, "manifest": manifest, "model": initial.state_dict()}, checkpoint)
                args = SimpleNamespace(checkpoint=checkpoint, data_root=root, test_file=source, img_dir=images,
                    text_embeddings=text_file, prompt=text.PROMPT, k=1, seed=10, eval_seed=10, epoch=1,
                    lr=1e-3, batch=2, test_batch=2, workers=0, weight_decay=0., clip=12.5, init="mean",
                    device="cpu", gpu_ids=None, eval_repeat_num=1, greedy=True, max_batches=0,
                    eval_max_batches=0, log_root=root / "run", resume=False, prepare_only=False)
                def network_factory(config, manifest, state):
                    return TinyNetwork(config, manifest)
                with patch.object(engine, "parse_args", side_effect=lambda *a, **k: copy.deepcopy(args)), \
                     patch.object(engine, "make_network", side_effect=network_factory):
                    branch = TRANSFER / branch_name
                    engine.train_main(mode, branch)
                    last_path = args.log_root / "checkpoints/checkpoint.pth"
                    last = torch.load(last_path, weights_only=True)
                    for name, value in initial.state_dict().items():
                        if name.startswith("backbone.") or (mode == "embedding" and name != "subject_embed.weight"):
                            torch.testing.assert_close(last["model"][name], value, rtol=0, atol=0)
                    if mode == "embedding":
                        torch.testing.assert_close(last["model"]["subject_embed.weight"][:2], initial.subject_embed.weight, rtol=0, atol=0)
                    else:
                        self.assertFalse(torch.equal(last["model"]["head.weight"], initial.head.weight))
                    self.assertFalse(torch.equal(last["model"]["subject_embed.weight"][2], initial.subject_embed.weight.mean(0)))
                    self.assertEqual((last["report"]["support_samples"], last["report"]["test_samples"]), (2, 4))
                    args.resume, args.epoch = True, 2
                    engine.train_main(mode, branch)
                    last = torch.load(last_path, weights_only=True)
                    self.assertEqual(last["epoch"], 1)
                    self.assertEqual(len(last["report"]["history"]), 2)
                    with self.assertRaises(ValueError):
                        require_mode(last["manifest"], "embedding" if mode == "predictor" else "predictor")
                    args.lr = .1
                    with self.assertRaises(ValueError):
                        engine.train_main(mode, branch)
                    args.lr = 1e-3
                    args.checkpoint = args.log_root / "checkpoints/best.pth"
                    engine.test_main(mode, branch)
                    report = json.loads((args.log_root / "report.json").read_text())
                    predictions = report["reevaluation"]["predictions"]
                    self.assertEqual(len(predictions), 4)
                    selected = {record_identity(r) for r in json.loads((args.log_root / "support.json").read_text())}
                    self.assertFalse(selected & {record_identity(p) for p in predictions})
                    self.assertEqual(predictions[0]["original_size"], [800, 600])
                    for prediction in predictions:
                        for i, point in enumerate(prediction["scanpath"]):
                            self.assertAlmostEqual(prediction["X"][i], point[0] / 512 * 800)
                            self.assertAlmostEqual(prediction["Y"][i], point[1] / 352 * 600)
