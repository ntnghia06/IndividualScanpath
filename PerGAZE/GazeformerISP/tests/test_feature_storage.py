import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dataset.features import save_features, load_features, is_compressed, FEATURE_FORMAT


class FeatureStorageTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.directory = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.path = Path(self.directory.name) / "sample.pth"

    def tearDown(self):
        self.directory.cleanup()

    def test_batch_view_serializes_only_one_sample_and_losslessly_compresses(self):
        batch = torch.randn(4, 2048, 24, 32)
        batch[batch < 1] = 0
        save_features(self.path, "VQA/a.jpg", batch[2], 100, 200, compression="none")
        self.assertLess(self.path.stat().st_size, 7 * 2**20)
        first, width, height = load_features(self.path, "VQA/a.jpg")
        self.assertEqual(first.untyped_storage().nbytes(), first.numel() * 4)
        torch.testing.assert_close(first, batch[2], rtol=0, atol=0)
        save_features(self.path, "VQA/a.jpg", batch[2], 100, 200)
        self.assertTrue(is_compressed(self.path))
        second, *_ = load_features(self.path, "VQA/a.jpg")
        torch.testing.assert_close(second, batch[2], rtol=0, atol=0)
        self.assertEqual(second.dtype, torch.float32)
        self.assertEqual((width, height), (100, 200))
        self.assertLess(self.path.stat().st_size, 6 * 2**20)

    def test_old_uncompressed_batch_views_still_load_as_compact_tensors(self):
        batch = torch.zeros(4, 2048, 24, 32)
        torch.save({"format": FEATURE_FORMAT, "source": "VQA/a.jpg", "features": batch[1],
                    "width": 100, "height": 200}, self.path)
        self.assertGreater(self.path.stat().st_size, 24 * 2**20)
        feature, *_ = load_features(self.path, "VQA/a.jpg")
        self.assertEqual(feature.untyped_storage().nbytes(), 6 * 2**20)

    def test_write_failure_preserves_existing_cache_and_removes_partial_file(self):
        feature = torch.zeros(2048, 24, 32)
        save_features(self.path, "VQA/a.jpg", feature, 100, 200)
        original = self.path.read_bytes()
        def failed_save(payload, stream):
            stream.write(b"incomplete")
            raise RuntimeError("simulated full disk")
        with patch("torch.save", side_effect=failed_save):
            with self.assertRaisesRegex(OSError, "free disk"):
                save_features(self.path, "VQA/a.jpg", feature, 100, 200)
        self.assertEqual(self.path.read_bytes(), original)
        self.assertFalse(self.path.with_suffix(".tmp").exists())
