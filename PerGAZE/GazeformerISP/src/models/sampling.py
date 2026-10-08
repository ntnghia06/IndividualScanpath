import numpy as np
import torch
from geometry import SCANPATH_SIZE, ACTION_GRID, ACTION_COUNT


def sample_scanpaths(prediction, min_length=1, width=SCANPATH_SIZE[1], height=SCANPATH_SIZE[0], greedy=False):
    original_probabilities = prediction["all_actions_prob"]
    if not torch.isfinite(original_probabilities).all() or (original_probabilities < 0).any():
        raise FloatingPointError("Invalid action probabilities")
    probabilities = original_probabilities.detach().clone()
    if probabilities.shape[-1] != ACTION_COUNT:
        raise ValueError(f"Expected {ACTION_COUNT} actions for the PerGAZE Gazeformer grid")
    probabilities[:, :min_length, 0] = 0
    total = probabilities.sum(-1, keepdim=True)
    if (total <= 0).any():
        raise FloatingPointError("No action probability remains after masking STOP")
    probabilities = probabilities / total
    distribution = torch.distributions.Categorical(probs=probabilities)
    actions = probabilities.argmax(-1) if greedy else distribution.sample()
    mu, variance = prediction["log_normal_mu"], prediction["log_normal_sigma2"]
    # Reproduce the original sampler: sigma2 scales noise directly.
    if not torch.isfinite(mu).all() or not torch.isfinite(variance).all() or (variance <= 0).any():
        raise FloatingPointError("Nonfinite or nonpositive duration parameters")
    times = (mu if greedy else mu + torch.randn_like(mu) * variance).exp()
    if not torch.isfinite(times).all():
        raise FloatingPointError("Duration sampling produced nonfinite values")
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
            path.append(((cell % ACTION_GRID[1] + .5) * width / ACTION_GRID[1],
                         (cell // ACTION_GRID[1] + .5) * height / ACTION_GRID[0], float(duration)))
        paths.append(np.asarray(path, dtype=np.float64).reshape(-1, 3))
    selected_probability = original_probabilities.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    return paths, (selected_probability + 1e-7).log(), times, active, duration_mask
