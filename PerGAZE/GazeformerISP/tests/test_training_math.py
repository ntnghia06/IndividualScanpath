import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, Mock
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import train
from models.sampling import sample_scanpaths


class OriginalTrainingMathTest(unittest.TestCase):
    def test_raw_action_probability_used_after_stop_mask(self):
        count = 769 if "Gazeformer" in str(Path(__file__)) else 1201
        probabilities = torch.zeros(1, 1, count)
        probabilities[0, 0, :2] = torch.tensor([.75, .25])
        probabilities.requires_grad_(True)
        prediction = {"all_actions_prob": probabilities, "log_normal_mu": torch.zeros(1, 1),
                      "log_normal_sigma2": torch.ones(1, 1)}
        _, logp, *_ = sample_scanpaths(prediction, min_length=1, greedy=True)
        torch.testing.assert_close(logp, torch.log(torch.tensor([[.25 + 1e-7]])))
        logp.sum().backward()
        self.assertGreater(probabilities.grad[0, 0, 1].item(), 0.)
        self.assertEqual(probabilities[0, 0, 0].item(), .75)

    def test_overflow_raises_instead_of_clipping(self):
        count = 769 if "Gazeformer" in str(Path(__file__)) else 1201
        probabilities = torch.zeros(1, 1, count); probabilities[:, :, 1] = 1
        prediction = {"all_actions_prob": probabilities, "log_normal_mu": torch.full((1, 1), 100.),
                      "log_normal_sigma2": torch.ones(1, 1)}
        with self.assertRaisesRegex(FloatingPointError, "nonfinite"):
            sample_scanpaths(prediction, greedy=True)

    def test_rl_uses_batch_mask_denominators_and_sum_over_trials(self):
        active = torch.tensor([[1, 0, 0], [1, 1, 1]], dtype=torch.bool)
        mask = torch.tensor([[0, 0, 0], [1, 1, 0]], dtype=torch.bool)
        action = [torch.tensor([[-1., -2., -3.], [-4., -5., -6.]]),
                  torch.tensor([[-2., -1., -1.], [-2., -3., -4.]])]
        duration = torch.tensor([[-1., -2., -3.], [-3., -4., -5.]])
        samples = [(np.zeros((1, 3)), np.zeros((2, 3)))] * 2
        rewards = torch.tensor([[.2, .8], [.6, .1]])
        metrics = Mock(); metrics.reward.side_effect = rewards.flatten().tolist()
        args = SimpleNamespace(rl_sample_number=2, min_length=0, rl_baseline="mean")
        batch = {"images": torch.zeros(2, 1), "fix_vectors": [np.zeros((1, 3))] * 2,
                 "metadata": [{"condition": "present"}, {"condition": "vqa"}]}
        sampled = [(paths, logp, torch.ones(2, 3), active, mask) for paths, logp in zip(samples, action)]
        with patch.object(train, "forward", return_value={}), \
             patch.object(train, "sample_scanpaths", side_effect=sampled), \
             patch.object(train, "duration_log_prob", return_value=duration):
            loss, _ = train.rl_loss(torch.nn.Linear(1, 1), batch, args, metrics)
        logp = torch.stack([(a * active).sum(-1) / active.sum()
                            + (duration * mask).sum(-1) / mask.sum() for a in action])
        expected = -(logp * (rewards - rewards.mean(0, keepdim=True))).sum()
        torch.testing.assert_close(loss, expected)
