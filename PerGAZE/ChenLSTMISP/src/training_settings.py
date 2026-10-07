"""Restore saved training settings when resuming a run."""
TRAINING_KEYS = (
    "lr", "weight_decay", "rl_lr_initial_decay", "clip", "lambda_1",
    "embedding_dim", "action_map_num", "dropout", "rl_sample_number",
    "warmup_epoch", "test_batch", "eval_repeat_num", "blur_sigma",
    "supervised_save", "rl_baseline", "start_rl_epoch", "batch", "seed",
    "max_length", "min_length", "no_eval_epoch",
)


def restore_training_settings(args, saved):
    provided = set(args._provided_hyperparams)
    for key in TRAINING_KEYS:
        if key in saved:
            if key in provided and getattr(args, key) != saved[key]:
                raise ValueError(f"Cannot change {key} while resuming; start a new run")
            setattr(args, key, saved[key])
    if "epoch" not in provided:
        args.epoch = saved["epoch"]
