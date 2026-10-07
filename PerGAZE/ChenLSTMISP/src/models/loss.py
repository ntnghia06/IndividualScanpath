import math
import torch
import torch.nn.functional as F


def supervised_loss(prediction, batch, duration_weight=1.0):
    action = -(batch["target_scanpaths"] * F.log_softmax(prediction["actions"], -1)).sum(-1)
    action = (action * batch["action_masks"]).sum() / batch["action_masks"].sum().clamp_min(1)
    duration = -duration_log_prob(batch["durations"], prediction)
    duration = (duration * batch["duration_masks"]).sum() / batch["duration_masks"].sum().clamp_min(1)
    return action + duration_weight * duration, action, duration


def duration_log_prob(durations, prediction):
    variance = prediction["log_normal_sigma2"].clamp_min(1e-8)
    log_time = durations.clamp_min(1e-8).log()
    return -log_time - .5 * (math.log(2 * math.pi) + variance.log()) - (log_time - prediction["log_normal_mu"]) ** 2 / (2 * variance)
