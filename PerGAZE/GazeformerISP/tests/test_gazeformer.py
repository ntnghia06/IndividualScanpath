import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from contextlib import nullcontext
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
import numpy as np
from dataset.text import add_text_manifest, GazeformerPerGAZE
from dataset.schema import make_manifest
from models.gazeformer.model import GazeformerISP


class GazeformerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        cls.args = SimpleNamespace(hidden_dim=32, text_dim=16, nhead=4, num_encoder=1,
            num_decoder=1, encoder_dropout=0., decoder_dropout=0., embedding_dim=8,
            action_map_num=2, max_length=1, dropout=0., train_backbone=False, backbone_weights="coco")
        cls.manifest = {"subjects": {"coco:1": 0}, "text_vocabulary": {"<pad>": 0, "<unk>": 1, "car": 2, "chair": 3}}
        cls.model = GazeformerISP(cls.args, cls.manifest, pretrained=False).eval()
        cls.features = torch.randn(1, 300, 2048)

    def predict(self, mask, tokens, guidance=None):
        return self.model.forward_features(self.features, torch.tensor([0]),
            torch.zeros(1, 1, 30, 40) if guidance is None else guidance,
            torch.tensor([tokens]), torch.tensor([mask]))

    def test_batch_one_length_one_output(self):
        prediction = self.predict(1., [2, 0])
        self.assertEqual(prediction["all_actions_prob"].shape, (1, 1, 1201))
        self.assertEqual(prediction["log_normal_mu"].shape, (1, 1))
        torch.testing.assert_close(prediction["all_actions_prob"].sum(-1), torch.ones(1, 1))

    def test_absent_task_is_masked(self):
        a, b = self.predict(0., [2, 0]), self.predict(0., [3, 0])
        torch.testing.assert_close(a["all_actions_prob"], b["all_actions_prob"])

    def test_guidance_changes_prediction(self):
        a = torch.zeros(1, 1, 30, 40)
        b = a.clone()
        a[:, :, :15, :20] = 1
        b[:, :, 15:, 20:] = 1
        first = self.predict(1., [2, 0], a)["all_actions_prob"]
        second = self.predict(1., [2, 0], b)["all_actions_prob"]
        self.assertGreater(float((first - second).abs().max()), 1e-8)

    def test_external_vectors_and_frozen_backbone(self):
        expected = torch.ones(1, 16)
        torch.testing.assert_close(self.model.encode_text(torch.tensor([[2]]), torch.ones(1), expected), expected)
        self.model.train()
        self.assertFalse(self.model.backbone.training)
        self.assertTrue(all(not p.requires_grad for p in self.model.backbone.parameters()))
        self.model.eval()

    def test_vocabulary_excludes_validation_words(self):
        records = [{"name": f"{i}.jpg", "condition": "vqa", "subject": "AL", "task": f"word{i}"} for i in range(20)]
        manifest = make_manifest(records)
        add_text_manifest(records, manifest)
        for record in records:
            split = manifest["splits"]["air:" + record["name"]]
            self.assertEqual(record["task"] in manifest["text_vocabulary"], split == "train")

    def test_npz_vectors_route_by_task_and_mask_absent(self):
        records = [{"name": "a.jpg", "condition": c, "subject": 1, "task": "car"}
                   for c in ("present", "absent")]
        manifest = make_manifest(records)
        add_text_manifest(records, manifest)
        archive = {"tasks": np.array(["car"]), "vectors": np.ones((1, 16), dtype=np.float32)}
        with patch("numpy.load", return_value=nullcontext(archive)):
            data = GazeformerPerGAZE(records, manifest, ".", ".", split="all", text_embeddings="unused.npz")
        with patch("dataset.dataset.PerGAZE.__getitem__", side_effect=lambda index: {}):
            present, absent = data[0], data[1]
        torch.testing.assert_close(present["task_embeddings"], torch.ones(16))
        torch.testing.assert_close(absent["task_embeddings"], torch.zeros(16))
        self.assertEqual(float(absent["task_mask"]), 0.)


if __name__ == "__main__":
    unittest.main()
