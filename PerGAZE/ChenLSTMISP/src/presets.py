"""Hyperparameter values from the original AiR/COCO opts.py and train.sh."""
COMMON = {
    "lr": 1e-4, "weight_decay": 5e-5, "rl_lr_initial_decay": .1,
    "clip": 12.5, "lambda_1": 1., "embedding_dim": 128,
    "action_map_num": 4, "dropout": .2, "rl_sample_number": 5,
    "warmup_epoch": 1, "test_batch": 1, "eval_repeat_num": 1,
    "blur_sigma": None, "supervised_save": True, "rl_baseline": "mean",
}

def preset_defaults(name):
    base = name.removesuffix("_run")
    defaults = {**COMMON, **PRESETS[base]}
    if name.endswith("_run"):
        defaults.update(RUN_OVERRIDES[base])
    return defaults

PRESETS = {
    "air": {"epoch": 30, "start_rl_epoch": 20, "batch": 2, "seed": 1,
            "max_length": 16, "min_length": 0, "no_eval_epoch": -1},
    "coco": {"epoch": 30, "start_rl_epoch": 20, "batch": 3, "seed": 1,
             "max_length": 7, "min_length": 0, "no_eval_epoch": 1},
}
RUN_OVERRIDES = {
    "air": {"epoch": 40, "seed": 1},
    "coco": {"epoch": 20, "start_rl_epoch": 10, "seed": 10},
}


def restore_training_settings(args, saved):
    # Training hyperparameters stay fixed across resume; --epoch may extend a run.
    keys = (set(COMMON) | set(PRESETS["air"]) | {"hyperparam_preset"}) - {"epoch"}
    provided = set(args._provided_hyperparams)
    for key in keys:
        if key in saved:
            if key in provided and getattr(args, key) != saved[key]:
                raise ValueError(f"Cannot change {key} while resuming; start a new run")
            setattr(args, key, saved[key])
    if "epoch" not in provided:
        args.epoch = saved["epoch"]
