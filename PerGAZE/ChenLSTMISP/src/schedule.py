"""Original AiR/COCO iteration-based warmup and linear decay."""


def learning_rate_factor(iteration, args, batches_per_epoch):
    warmup = batches_per_epoch * args.warmup_epoch
    supervised = batches_per_epoch * args.start_rl_epoch
    if warmup > 0 and iteration <= warmup:
        return iteration / warmup
    if iteration <= supervised:
        return max(0., 1. - (iteration - warmup) / max(1, supervised - warmup))
    rl_steps = batches_per_epoch * (args.epoch - args.start_rl_epoch)
    if rl_steps <= 0:
        return 0.
    return args.rl_lr_initial_decay * max(0., 1. - (iteration - supervised) / rl_steps)
