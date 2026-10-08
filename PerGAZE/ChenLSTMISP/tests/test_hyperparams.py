import ast
import re
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from opts import parse_opt
from training_settings import TRAINING_KEYS, restore_training_settings
from schedule import learning_rate_factor


class HyperparameterTest(unittest.TestCase):
    def test_defaults_keep_air_hyperparameters_with_requested_schedule(self):
        model_root = Path(__file__).resolve().parents[1]
        source = model_root.parents[1] / "AiR" / model_root.name
        original = {}
        for node in ast.walk(ast.parse((source / "src/opts.py").read_text())):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument" and node.args:
                name = ast.literal_eval(node.args[0]).lstrip("-")
                default = next((kw.value for kw in node.keywords if kw.arg == "default"), None)
                if default is not None:
                    original[name] = ast.literal_eval(default)
        for key, value in re.findall(r"--(seed|epoch|start_rl_epoch)\s+(\d+)", (source / "bash/train.sh").read_text()):
            original[key] = int(value)
        if model_root.name == "GazeformerISP":
            original["dropout"] = original["cls_dropout"]
            original["embedding_dim"] = original["subject_feature_dim"]
        original.update(epoch=10, start_rl_epoch=5, no_eval_epoch=-1)
        with patch.object(sys, "argv", ["train.py"]):
            args = parse_opt()
        for key in set(TRAINING_KEYS) | {"epoch"}:
            if key in original:
                self.assertEqual(getattr(args, key), original[key], key)
        self.assertFalse(hasattr(args, "hyperparam_preset"))

    def test_five_supervised_and_five_rl_epochs(self):
        with patch.object(sys, "argv", ["train.py"]):
            args = parse_opt()
        self.assertEqual(args.epoch, 10)
        self.assertEqual([i for i in range(args.epoch) if i < args.start_rl_epoch], list(range(5)))
        self.assertEqual([i for i in range(args.epoch) if i >= args.start_rl_epoch], list(range(5, 10)))
        self.assertEqual(args.warmup_epoch, 1)
        self.assertEqual(args.no_eval_epoch, -1)
        self.assertEqual(learning_rate_factor(10, args, 10), 1.)
        self.assertEqual(learning_rate_factor(50, args, 10), 0.)
        self.assertAlmostEqual(learning_rate_factor(75, args, 10), .05)
        self.assertEqual(learning_rate_factor(100, args, 10), 0.)

    def test_cli_can_override_direct_defaults(self):
        with patch.object(sys, "argv", ["train.py", "--batch", "4", "--epoch", "45"]):
            args = parse_opt()
        self.assertEqual(args.epoch, 45)
        self.assertEqual(args.batch, 4)
        self.assertEqual(args.max_length, 16)
        self.assertIsNone(args.blur_sigma)
        self.assertEqual(args.test_batch, 1)

    def test_warmup_decay_matches_source_formula(self):
        args = SimpleNamespace(warmup_epoch=1, start_rl_epoch=20, epoch=30, rl_lr_initial_decay=.1)
        for step in (0, 5, 10, 55, 100, 199, 200, 201, 250, 300):
            if step <= 10:
                expected = step / 10
            elif step <= 200:
                expected = 1 - (step - 10) / 190
            else:
                expected = .1 * (1 - (step - 200) / 100)
            self.assertAlmostEqual(learning_rate_factor(step, args, 10), expected)

    def test_resume_restores_saved_settings_and_keeps_requested_target_epoch(self):
        with patch.object(sys, "argv", ["train.py"]):
            saved = vars(parse_opt()).copy()
        saved["max_length"] = 7
        with patch.object(sys, "argv", ["train.py", "--epoch", "45"]):
            args = parse_opt()
        restore_training_settings(args, saved)
        self.assertEqual(args.max_length, 7)
        self.assertEqual(args.epoch, 45)

    def test_scheduler_state_resume_matches_continuous_run(self):
        args = SimpleNamespace(warmup_epoch=1, start_rl_epoch=2, epoch=4, rl_lr_initial_decay=.1)
        parameter = torch.nn.Parameter(torch.ones(1))
        optimizer = torch.optim.Adam([parameter], lr=1e-4)
        schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: learning_rate_factor(step, args, 3))
        for _ in range(5):
            optimizer.step()
            schedule.step()
        saved_opt, saved_schedule = optimizer.state_dict(), schedule.state_dict()
        resumed = torch.optim.Adam([torch.nn.Parameter(torch.ones(1))], lr=1e-4)
        resumed.load_state_dict(saved_opt)
        resumed_schedule = torch.optim.lr_scheduler.LambdaLR(resumed, lambda step: learning_rate_factor(step, args, 3))
        resumed_schedule.load_state_dict(saved_schedule)
        for group, rate in zip(resumed.param_groups, resumed_schedule.get_last_lr()):
            group["lr"] = rate
        optimizer.step()
        schedule.step()
        resumed.step()
        resumed_schedule.step()
        self.assertEqual(schedule.get_last_lr(), resumed_schedule.get_last_lr())


if __name__ == "__main__":
    unittest.main()
