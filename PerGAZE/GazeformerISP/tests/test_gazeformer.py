import sys
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dataset.text import GazeformerPerGAZE, task_text
from dataset.schema import make_manifest
from dataset.features import cache_path, FEATURE_FORMAT
from models.gazeformer.model import GazeformerISP
from runtime import restore_config


class GazeformerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        cls.args = SimpleNamespace(hidden_dim=32, text_dim=768, nhead=4, num_encoder=1,
            num_decoder=1, encoder_dropout=0., decoder_dropout=0., embedding_dim=8,
            action_map_num=2, max_length=2, dropout=0.)
        cls.model = GazeformerISP(cls.args, {"subjects": {"tp:1": 0}}, pretrained=False).eval()
        cls.features = torch.randn(1, 2048, 24, 32)

    def predict(self, text, guidance=None):
        return self.model(self.features, torch.tensor([0]), guidance,
                          torch.tensor([[0]]), torch.tensor([1.]), text)

    def test_grid_and_text_conditioning_without_external_guidance(self):
        a = self.predict(torch.zeros(1, 768), torch.zeros(1, 1, 24, 32))
        b = self.predict(torch.ones(1, 768), torch.zeros(1, 1, 24, 32))
        c = self.predict(torch.zeros(1, 768), torch.ones(1, 1, 24, 32))
        self.assertEqual(a["all_actions_prob"].shape, (1, 2, 769))
        torch.testing.assert_close(a["all_actions_prob"].sum(-1), torch.ones(1, 2))
        torch.testing.assert_close(a["all_actions_prob"], c["all_actions_prob"], rtol=0, atol=0)
        self.assertGreater(float((a["all_actions_prob"] - b["all_actions_prob"]).abs().max()), 1e-8)
        self.assertFalse(any("backbone" in name or "word_embed" in name for name, _ in self.model.named_parameters()))

    def test_supervised_backward_from_cached_features(self):
        self.model.train()
        result = self.predict(torch.randn(1, 768))
        (result["actions"].square().mean() + result["log_normal_mu"].square().mean()).backward()
        self.assertTrue(torch.isfinite(self.model.subject_embed.weight.grad).all())
        self.model.zero_grad(set_to_none=True)
        self.model.eval()

    def test_cached_dataset_ta_uses_instruction_and_never_reads_attention_map(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            root = Path(tmp)
            rows = [{"name": "a.jpg", "condition": condition, "subject": 1, "task": "car",
                     "X": [5], "Y": [5], "T": [200], "bbox": [1, 1, 2, 2]}
                    for condition in ("present", "absent")]
            for row in rows:
                folder = root / {"present": "TP", "absent": "TA"}[row["condition"]]
                folder.mkdir()
                Image.new("RGB", (10, 10)).save(folder / "a.jpg")
                target, relative = cache_path(root, root, row)
                torch.save({"features": torch.ones(2048, 24, 32, dtype=torch.float16),
                            "format": FEATURE_FORMAT, "source": relative, "width": 10, "height": 10}, target)
            texts = [task_text(row) for row in rows]
            np.savez(root / "text.npz", tasks=np.asarray(texts), vectors=np.stack([
                np.ones(768), np.full(768, 2)]).astype(np.float32))
            data = GazeformerPerGAZE(rows, make_manifest(rows), root, "missing_attention", split="all",
                                    feature_dir=root, text_embeddings=root / "text.npz")
            with patch("dataset.dataset.Image.open", side_effect=AssertionError("No decoding during train")):
                tp, ta = data[0], data[1]
            self.assertEqual(tuple(ta["images"].shape), (2048, 24, 32))
            self.assertEqual(ta["images"].dtype, torch.float32)
            self.assertEqual(ta["task_mask"].item(), 1.)
            torch.testing.assert_close(ta["task_embeddings"], torch.full((768,), 2.))
            self.assertEqual(ta["attention_maps"].sum().item(), 0.)
            self.assertNotEqual(tp["subjects"].item(), ta["subjects"].item())
            self.assertEqual(texts[1], "Search for the car in the image.")

    def test_old_checkpoint_geometry_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "geometry"):
            restore_config(SimpleNamespace(), {"manifest": {}}, {})
