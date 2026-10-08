import sys
import json
import unittest
from pathlib import Path
from unittest.mock import patch, Mock
from types import SimpleNamespace
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from utils.evaluation import Metrics, summarize, retrieval
from models.loss import duration_log_prob, supervised_loss
from models.sampling import sample_scanpaths
import train


class OriginalMetricsTest(unittest.TestCase):
    def test_multimatch_padding_count_and_json_safe_invalid_scores(self):
        target = np.array([[10., 20., .2]])
        prediction = np.empty((0, 3))
        metrics = Metrics()
        with patch("multimatch_gaze.docomparison", return_value=np.array([1., np.nan, 1., 1., 1.])) as compare:
            result = metrics.pair(target, prediction)
        first, second = compare.call_args.args
        self.assertEqual(len(first), 3); self.assertEqual(len(second), 3)
        self.assertEqual(tuple(first[1]), (1., 1., .001))
        for name in ("vector", "direction", "length", "position", "duration"):
            self.assertIsNone(result["MultiMatch_" + name])
        rows = [{"condition": "present", "subject_key": "tp:7", "metrics": result}]
        report = summarize(rows)
        self.assertEqual(report["overall"]["multimatch_padding"]["pairs"], 1)
        self.assertEqual(report["overall"]["multimatch_padding"]["targets"], 1)
        self.assertEqual(report["overall"]["multimatch_padding"]["predictions"], 1)
        for name in ("vector", "direction", "length", "position", "duration"):
            self.assertEqual(report["overall"]["metrics"]["MultiMatch_" + name]["valid_count"], 0)
        json.dumps(report, allow_nan=False)

    def test_retrieval_groups_conditions_and_questions_independently(self):
        rows, targets = [], {}
        for condition, prefix, qid in (("present", "tp", None), ("absent", "ta", None),
                                      ("vqa", "air", "q1"), ("vqa", "air", "q2")):
            for subject in (7, 8):
                index = len(rows)
                path = np.array([[subject, 10, .2], [subject + 1, 20, .3], [subject + 2, 30, .4]])
                targets[index] = path
                rows.append({"condition": condition, "name": "same.jpg", "task": "car",
                    "qid": qid, "repeat": 0, "subject_key": f"{prefix}:{subject}",
                    "record_index": index, "scanpath": path.tolist()})
        metrics = Mock()
        metrics.retrieval_score.side_effect = lambda gt, pred: .9 if gt[0, 0] == pred[0, 0] else .1
        result = retrieval(rows, targets, metrics)
        self.assertEqual(len(result["groups"]), 4)
        self.assertEqual(result["overall"]["R@1"], 100.)
        self.assertEqual(result["overall"]["MRR"], 1.)
        self.assertEqual(result["by_condition"]["vqa"]["queries"], 4)
        self.assertTrue(all(group["candidate_count"] == 2 for group in result["groups"]))
        self.assertEqual(result["groups"][0]["ScanMatch_with_duration_matrix"], [[.9, .1], [.1, .9]])

    def test_duration_loss_matches_original_formula_and_mask(self):
        times = torch.tensor([[.2, .3, 0.]])
        mu = torch.tensor([[.1, -.2, .3]], requires_grad=True)
        variance = torch.tensor([[.8, 1.2, 2.]], requires_grad=True)
        expected = torch.log(1 / (times + 1e-7) / torch.sqrt(2 * np.pi * variance)) \
            - (torch.log(times + 1e-7) - mu).square() / (2 * variance)
        prediction = {"log_normal_mu": mu, "log_normal_sigma2": variance,
                      "actions": torch.zeros(1, 3, 2)}
        actual = duration_log_prob(times, prediction)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        batch = {"durations": times, "duration_masks": torch.tensor([[1., 1., 0.]]),
                 "target_scanpaths": torch.zeros(1, 3, 2), "action_masks": torch.ones(1, 3)}
        _, _, duration = supervised_loss(prediction, batch)
        torch.testing.assert_close(duration, -expected[0, :2].mean())

    def test_rl_clips_loss_only_and_resamples_nan_reward(self):
        args = SimpleNamespace(rl_sample_number=2, min_length=0, rl_baseline="mean")
        batch = {"images": torch.zeros(2, 1), "fix_vectors": [np.zeros((3, 3))] * 2,
                 "metadata": [{"condition": "present"}, {"condition": "vqa"}]}
        times = torch.tensor([[6.], [140.]])
        sample = ([np.ones((3, 3))] * 2, torch.ones(2, 1), times,
                  torch.ones(2, 1, dtype=torch.bool), torch.ones(2, 1, dtype=torch.bool))
        metrics = Mock(); metrics.reward.side_effect = [float("nan"), .1, .2, .8, .6, .1]
        with patch.object(train, "forward", return_value={}), \
             patch.object(train, "sample_scanpaths", return_value=sample) as sampler, \
             patch.object(train, "duration_log_prob", return_value=torch.ones(2, 1)) as log_duration:
            loss, _ = train.rl_loss(torch.nn.Linear(1, 1), batch, args, metrics)
        self.assertEqual(sampler.call_count, 3)
        self.assertEqual(log_duration.call_count, 2)
        cap = 100. if "Gazeformer" in str(Path(__file__)) else 3.
        torch.testing.assert_close(log_duration.call_args.args[0], torch.tensor([[min(6., cap)], [100.]]))
        torch.testing.assert_close(times, torch.tensor([[6.], [140.]]))
        self.assertTrue(torch.isfinite(loss))

    def test_scanmatch_bins_match_original(self):
        metrics = Metrics()
        # Actual similarity against itself verifies temporal and spatial matchers.
        path = np.array([[10., 20., .2], [40., 60., .3], [100., 120., .4]])
        self.assertEqual(metrics.scanmatch(path, path), (1., 1.))

    def test_zero_sampled_duration_uses_original_epsilon_formula(self):
        count = 769 if "Gazeformer" in str(Path(__file__)) else 1201
        probability = torch.zeros(1, 1, count); probability[:, :, 1] = 1
        prediction = {"all_actions_prob": probability, "log_normal_mu": torch.full((1, 1), -1000.),
                      "log_normal_sigma2": torch.ones(1, 1)}
        _, _, times, *_ = sample_scanpaths(prediction, greedy=True)
        self.assertEqual(times.item(), 0.)
        self.assertTrue(torch.isfinite(duration_log_prob(times, prediction)).all())

    def test_retrieval_does_not_discard_scanmatch_when_multimatch_is_nan(self):
        path = np.array([[10., 20., .2], [40., 60., .3], [100., 120., .4]])
        metrics = Metrics()
        with patch.object(metrics, "multimatch", side_effect=AssertionError("Retrieval must not depend on MM")):
            self.assertEqual(metrics.retrieval_score(path, path), 1.)
        # Reward rejection remains separate from the eval retrieval behavior.
        with patch.object(metrics, "multimatch", return_value=np.full(5, np.nan)):
            self.assertTrue(np.isnan(metrics.reward(path, path)))

    def test_supervised_action_loss_matches_original_at_tiny_probabilities(self):
        logits = torch.tensor([[[0., -80.], [2., -3.]]], requires_grad=True)
        target = torch.tensor([[[0., 1.], [1., 0.]]])
        mask = torch.tensor([[1., 0.]])
        prediction = {"actions": logits, "log_normal_mu": torch.zeros(1, 2),
                      "log_normal_sigma2": torch.ones(1, 2)}
        batch = {"target_scanpaths": target, "action_masks": mask,
                 "durations": torch.full((1, 2), .2), "duration_masks": torch.ones(1, 2)}
        expected = -(target * torch.log(torch.softmax(logits, -1) + 1e-7) * mask[..., None]).sum() / mask.sum()
        _, actual, _ = supervised_loss(prediction, batch)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(torch.autograd.grad(actual, logits, retain_graph=True)[0],
                                   torch.autograd.grad(expected, logits)[0], rtol=0, atol=0)
        self.assertLess(actual.item(), 17.)  # log_softmax would produce 80 here.
