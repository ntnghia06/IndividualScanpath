import numpy as np
import torch


def sample_scanpaths(prediction, min_length=1, width=320, height=240, greedy=False):
    probabilities = prediction["all_actions_prob"].clone()
    probabilities[:, :min_length, 0] = 0
    probabilities = probabilities / probabilities.sum(-1, keepdim=True).clamp_min(1e-8)
    distribution = torch.distributions.Categorical(probs=probabilities)
    actions = probabilities.argmax(-1) if greedy else distribution.sample()
    mu, variance = prediction["log_normal_mu"], prediction["log_normal_sigma2"]
    # The model predicts variance, so sampling uses its square root.
    times = (mu if greedy else mu + torch.randn_like(mu) * variance.sqrt()).clamp(-10, 10).exp()
    active = torch.ones_like(actions, dtype=torch.bool)
    if actions.shape[1] > 1:
        active[:, 1:] = (actions[:, :-1] == 0).cumsum(1) == 0
    duration_mask = active & (actions != 0)
    paths = []
    for sample_actions, durations in zip(actions.detach().cpu().numpy(), times.detach().cpu().numpy()):
        path = []
        for action, duration in zip(sample_actions, durations):
            if action == 0:
                break
            cell = int(action) - 1
            path.append(((cell % 40 + .5) * width / 40, (cell // 40 + .5) * height / 30, float(duration)))
        paths.append(np.asarray(path, dtype=np.float64).reshape(-1, 3))
    return paths, distribution.log_prob(actions), times, active, duration_mask
