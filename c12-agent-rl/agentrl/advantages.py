"""Advantages: how much better than expected was this trajectory, or this action.

Trajectory level (GRPO). Sample G trajectories for the same task. Each gets
    A_i = (R_i - mean(R_1..R_G)) / (std(R_1..R_G) + eps)
and every action token of trajectory i receives the same A_i. No critic. If all G rewards are equal, every A_i is
0 and the group teaches nothing.

Action level (critic + GAE). A value head estimates V(s_t), the expected final reward from state s_t. The TD error
    delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
says whether action t made things better or worse than expected; GAE sums them with decay gamma * lambda.
A wasted retry in a successful episode can then get a negative advantage even though the episode succeeded.
"""

from __future__ import annotations

import torch


def group_relative(rewards: torch.Tensor, groups: torch.Tensor, scale: bool = True,
                   eps: float = 1e-4) -> torch.Tensor:
    """rewards (N,), groups (N,) integer group ids -> advantages (N,).

    ``std`` is the unbiased sample std, as in TRL's GRPOTrainer. ``scale=False`` keeps only the mean baseline
    (the Dr. GRPO variant), which stops low-variance groups from being amplified.
    """
    adv = torch.zeros_like(rewards)
    for g in groups.unique():
        m = groups == g
        r = rewards[m]
        adv[m] = (r - r.mean()) / (r.std() + eps) if scale else r - r.mean()
    return adv


def gae(rewards: list[float], values: list[float], gamma: float = 1.0, lam: float = 0.95) -> list[float]:
    """Per-step advantages for one episode. ``values[t]`` = V(s_t); the episode ends after the last step."""
    adv, running = [0.0] * len(rewards), 0.0
    for t in reversed(range(len(rewards))):
        next_v = values[t + 1] if t + 1 < len(values) else 0.0
        delta = rewards[t] + gamma * next_v - values[t]
        running = delta + gamma * lam * running
        adv[t] = running
    return adv


def to_tokens(step_adv: list[list[float]], step_of_token: torch.Tensor) -> torch.Tensor:
    """Scatter per-step advantages onto action tokens. step_of_token (B, T), -1 off actions -> (B, T)."""
    out = torch.zeros(step_of_token.shape)
    for i, adv in enumerate(step_adv):
        idx = step_of_token[i]
        on = idx >= 0
        out[i, on] = torch.tensor(adv)[idx[on]]
    return out
