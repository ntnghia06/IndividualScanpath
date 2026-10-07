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
from presets import preset_defaults, restore_training_settings
from schedule import learning_rate_factor


class HyperparameterTest(unittest.TestCase):
    def test_presets_match_original_opts(self):
        model_root = Path(__file__).resolve().parents[1]
        architecture = model_root.name
        individual_root = model_root.parents[1]
        for preset, dataset in (("air", "AiR"), ("coco", "COCO_Search18")):
            tree = ast.parse((individual_root / dataset / architecture / "src/opts.py").read_text())
            original = {}
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument" and node.args:
                    name = ast.literal_eval(node.args[0]).lstrip("-")
                    default = next((kw.value for kw in node.keywords if kw.arg == "default"), None)
                    if default is not None:
                        original[name] = ast.literal_eval(default)
            defaults = preset_defaults(preset)
            if architecture == "GazeformerISP":
                original["dropout"] = original["cls_dropout"]
                original["embedding_dim"] = original["subject_feature_dim"]
            for key in defaults.keys() & original.keys():
                self.assertEqual(defaults[key], original[key], f"{architecture} {preset} {key}")

    def test_run_presets_match_original_shell_overrides(self):
        model_root = Path(__file__).resolve().parents[1]
        for name, dataset in (("air", "AiR"), ("coco", "COCO_Search18")):
            script = (model_root.parents[1] / dataset / model_root.name / "bash/train.sh").read_text()
            settings = preset_defaults(name + "_run")
            for key, value in re.findall(r"--(seed|epoch|start_rl_epoch)\s+(\d+)", script):
                self.assertEqual(settings[key], int(value))

    def test_cli_overrides_preset(self):
        with patch.object(sys, "argv", ["train.py", "--hyperparam_preset", "coco", "--batch", "4"]):
            args = parse_opt()
        self.assertEqual(args.max_length, 7)
        self.assertEqual(args.batch, 4)
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

    def test_resume_restores_schedule_and_keeps_requested_target_epoch(self):
        saved = {**preset_defaults("coco"), "hyperparam_preset": "coco"}
        args = SimpleNamespace(**preset_defaults("air"), hyperparam_preset="air", _provided_hyperparams=["epoch"])
        args.epoch = 45
        restore_training_settings(args, saved)
        self.assertEqual(args.max_length, 7)
        self.assertEqual(args.hyperparam_preset, "coco")
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
