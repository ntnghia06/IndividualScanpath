import math
import torch
import torch.nn.functional as F


def supervised_loss(prediction, batch, duration_weight=1.0):
    probabilities = F.softmax(prediction["actions"], dim=-1)
    action = -(batch["target_scanpaths"] * torch.log(probabilities + 1e-7)
               * batch["action_masks"].unsqueeze(-1)).sum() / batch["action_masks"].sum()
    duration = -duration_log_prob(batch["durations"], prediction)
    duration = duration[batch["duration_masks"] == 1].sum() / batch["duration_masks"].sum()
    return action + duration_weight * duration, action, duration


def duration_log_prob(durations, prediction):
    variance = prediction["log_normal_sigma2"]
    if not torch.isfinite(variance).all() or (variance <= 0).any():
        raise FloatingPointError("Nonfinite or nonpositive duration variance")
    epsilon = 1e-7
    return torch.log(1 / (durations + epsilon) * 1 / torch.sqrt(2 * math.pi * variance)) \
        - (torch.log(durations + epsilon) - prediction["log_normal_mu"]) ** 2 / (2 * variance)
