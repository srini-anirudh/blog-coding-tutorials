"""Returns, TD errors and GAE, written for (B, T) tensors.

    delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
    A_t^GAE = sum_l (gamma * lambda)^l * delta_{t+l}      computed backwards:  A_t = delta_t + gamma*lambda*A_{t+1}

lambda = 0 is one step of reality, then trust the critic. lambda = 1 is all of reality minus a baseline.
"""

from __future__ import annotations

import torch


def terminal_rewards(R: torch.Tensor, T: int, lengths: torch.Tensor | None = None) -> torch.Tensor:
    """Put the whole reward on the last real token, zeros elsewhere: (B,) -> (B, T)."""
    r = torch.zeros(R.shape[0], T, dtype=torch.float32)
    last = (lengths - 1) if lengths is not None else torch.full((R.shape[0],), T - 1)
    r[torch.arange(R.shape[0]), last] = R.float()
    return r


def returns(rewards: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    """G_t = sum_{k >= t} gamma^(k-t) r_k. With gamma = 1 and a terminal reward, every G_t equals R."""
    G = torch.zeros_like(rewards)
    running = torch.zeros(rewards.shape[0])
    for t in reversed(range(rewards.shape[1])):
        running = rewards[:, t] + gamma * running
        G[:, t] = running
    return G


def gae(rewards: torch.Tensor, values: torch.Tensor, gamma: float = 1.0, lam: float = 0.95,
        mask: torch.Tensor | None = None) -> torch.Tensor:
    """rewards, values: (B, T), values[:, t] = V(s_t). The value after the last real token is 0 (episode over).

    `mask` (B, T) marks real tokens. Positions after the end are skipped, so padding never leaks into the scan.
    """
    B, T = rewards.shape
    m = torch.ones(B, T) if mask is None else mask.float()
    adv = torch.zeros(B, T)
    running = torch.zeros(B)
    next_v = torch.zeros(B)
    for t in reversed(range(T)):
        delta = rewards[:, t] + gamma * next_v - values[:, t]
        running = delta + gamma * lam * running
        running = running * m[:, t]  # a padded position contributes nothing and resets the scan
        adv[:, t] = running
        next_v = values[:, t] * m[:, t]
    return adv
