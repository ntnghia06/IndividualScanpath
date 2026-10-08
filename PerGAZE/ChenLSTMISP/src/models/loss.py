import math
import torch
import torch.nn.functional as F


def supervised_loss(prediction, batch, duration_weight=1.0):
    action = -(batch["target_scanpaths"] * F.log_softmax(prediction["actions"], -1)).sum(-1)
    action = (action * batch["action_masks"]).sum() / batch["action_masks"].sum()
    duration = -duration_log_prob(batch["durations"], prediction)
    duration = (duration * batch["duration_masks"]).sum() / batch["duration_masks"].sum()
    return action + duration_weight * duration, action, duration


def duration_log_prob(durations, prediction):
    variance = prediction["log_normal_sigma2"]
    if not torch.isfinite(variance).all() or (variance <= 0).any():
        raise FloatingPointError("Nonfinite or nonpositive duration variance")
    log_time = (durations + 1e-7).log()
    return -log_time - .5 * (math.log(2 * math.pi) + variance.log()) - (log_time - prediction["log_normal_mu"]) ** 2 / (2 * variance)
