import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from PIL import Image

from dataset.dataset import PerGAZE, load_guidance
from dataset.schema import make_manifest, make_file_manifest, image_key, subject_key, split_counts
from models.sampling import sample_scanpaths
from utils.evaluation import Metrics
from utils.evaltools.visual_attention_metrics import _scanpath_to_string


class PerGAZETest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.root = Path(self.temporary.name)
        for folder in ("TP/bottle", "TA/bottle", "VQA", "attention"):
            (self.root / folder).mkdir(parents=True)
        for folder in ("TP/bottle", "TA/bottle", "VQA"):
            Image.new("RGB", (200, 100)).save(self.root / folder / "test.jpg")
        np.save(self.root / "attention/00001.npy", np.ones((20, 40)))
        self.records = [{"condition": c, "name": "test.jpg", "task": "bottle", "subject": s,
                         "X": [100., 200.], "Y": [50., -10.], "T": [200., 100.],
                         **({"bbox": [50, 25, 50, 25]} if c == "present" else {}),
                         **({"qid": "00001"} if c == "vqa" else {})}
                        for c, s in (("present", 1), ("absent", 1), ("vqa", "AL"))]

    def tearDown(self):
        self.temporary.cleanup()

    def test_guidance_routing_and_bbox_xywh(self):
        present, absent, vqa = [load_guidance(r, self.root / "attention", (200, 100)) for r in self.records]
        self.assertEqual(present.shape, (1, 22, 32))
        self.assertEqual(float(present[0, 8, 12]), 1.)
        self.assertEqual(float(present[0, 20, 25]), 0.)
        self.assertEqual(float(absent.max()), 0.)
        self.assertEqual(float(vqa.min()), 1.)

    def test_split_and_subject_namespaces(self):
        manifest = make_manifest(self.records)
        self.assertEqual(manifest, make_manifest(list(reversed(self.records))))
        self.assertEqual(image_key(self.records[0]), image_key(self.records[1]))
        self.assertEqual(subject_key(self.records[0]), subject_key(self.records[1]))
        self.assertNotEqual(subject_key(self.records[0]), subject_key(self.records[2]))

    def test_original_pixels_duration_stop_and_clipping(self):
        manifest = make_manifest(self.records)
        data = PerGAZE(self.records, manifest, self.root, self.root / "attention", split="all", max_length=4, blur_sigma=0)
        row = data[0]
        self.assertEqual(tuple(row["images"].shape), (3, 768, 1024))
        self.assertEqual(tuple(row["target_scanpaths"].shape), (4, 705))
        np.testing.assert_allclose(row["fix_vectors"][0], [256, 176, .2])
        self.assertLess(row["fix_vectors"][1, 0], 512)
        self.assertEqual(row["fix_vectors"][1, 1], 0)
        self.assertEqual(row["target_scanpaths"][0].argmax(), 1 + 11 * 32 + 16)
        self.assertEqual(row["target_scanpaths"][2, 0], 1)
        self.assertEqual(row["action_masks"].tolist(), [1, 1, 1, 0])
        self.assertEqual(row["duration_masks"].tolist(), [1, 1, 0, 0])

    def test_explicit_files_keep_same_image_in_requested_splits(self):
        train_rows = [self.records[0], self.records[2]]
        val_rows = [self.records[1]]
        manifest = make_file_manifest(train_rows, val_rows)
        rows = train_rows + val_rows
        train = PerGAZE(rows, manifest, self.root, self.root / "attention", split="train")
        val = PerGAZE(rows, manifest, self.root, self.root / "attention", split="validation")
        self.assertEqual([r for _, r in train.records], train_rows)
        self.assertEqual([r for _, r in val.records], val_rows)
        self.assertEqual(split_counts(rows, manifest)["validation"], {"absent": 1})
        self.assertEqual(manifest["record_splits"], ["train", "train", "validation"])

    def test_validation_observers_must_be_seen_in_train(self):
        with self.assertRaisesRegex(ValueError, "observers"):
            make_file_manifest([self.records[0]], [self.records[2]])

    def test_termination_masks(self):
        probability = torch.zeros(2, 3, 705)
        probability[0, :, 0] = 1  # immediate stop
        probability[1, 0, 1] = 1
        probability[1, 1:, 0] = 1
        prediction = {"all_actions_prob": probability, "log_normal_mu": torch.zeros(2, 3),
                      "log_normal_sigma2": torch.ones(2, 3)}
        paths, _, _, active, duration = sample_scanpaths(prediction, min_length=0, greedy=True)
        self.assertEqual([len(p) for p in paths], [0, 1])
        self.assertEqual(active.tolist(), [[True, False, False], [True, True, False]])
        self.assertEqual(duration.tolist(), [[False, False, False], [True, False, False]])

    def test_duration_sampling_uses_standard_deviation(self):
        probability = torch.zeros(1, 1, 705)
        probability[:, :, 1] = 1
        prediction = {"all_actions_prob": probability, "log_normal_mu": torch.zeros(1, 1),
                      "log_normal_sigma2": torch.full((1, 1), 4.)}
        with patch("torch.randn_like", side_effect=torch.ones_like):
            _, _, times, _, _ = sample_scanpaths(prediction)
        self.assertAlmostEqual(float(times[0, 0]), float(np.exp(2)), places=5)

    def test_sampling_frame_corners(self):
        probability = torch.zeros(1, 2, 705)
        probability[0, 0, 1] = 1
        probability[0, 1, 704] = 1
        prediction = {"all_actions_prob": probability, "log_normal_mu": torch.zeros(1, 2),
                      "log_normal_sigma2": torch.ones(1, 2)}
        paths, *_ = sample_scanpaths(prediction, greedy=True)
        np.testing.assert_allclose(paths[0][:, :2], [[8, 8], [504, 344]])

    def test_sed_frame_edges_stay_in_last_region(self):
        path = np.array([[504., 344.], [511.99, 351.99]])
        self.assertEqual(_scanpath_to_string(path, 352, 512, 5), "yy")

    def test_evaluation_identity_and_empty_prediction(self):
        target = np.array([[10., 20., .2], [40., 60., .3], [100., 120., .4]])
        metrics = Metrics()
        scores = metrics.pair(target, target.copy())
        self.assertAlmostEqual(scores["ScanMatch_without_duration"], 1.)
        self.assertAlmostEqual(scores["ScanMatch_with_duration"], 1.)
        self.assertEqual(scores["SED"], 0.)
        self.assertAlmostEqual(scores["STDE"], 1.)
        self.assertEqual(metrics.reward(target, np.empty((0, 3))), 0.)
        outlier = target.copy()
        outlier[:, 2] = 1000.
        rejected = metrics.pair(target, outlier)
        self.assertEqual(rejected["duration_outlier"], 1.)
        self.assertEqual(rejected["ScanMatch_with_duration"], 0.)


if __name__ == "__main__":
    unittest.main()
