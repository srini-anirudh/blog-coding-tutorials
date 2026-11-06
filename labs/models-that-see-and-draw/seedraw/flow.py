"""Rectified flow, written out as in the post.

    x_t   = (1 - t) x_0 + t eps,          t in [0, 1]
    L     = E || v(x_t, t, c) - (eps - x_0) ||^2
    step  x_{t - dt} = x_t - dt * v(x_t, t, c),   from t = 1 down to 0
"""

from __future__ import annotations

import math

import torch
from torch import nn


def time_features(t: torch.Tensor, dim: int = 16) -> torch.Tensor:
    """Sinusoidal features of t in [0, 1], shape (n, dim)."""
    freqs = torch.exp(torch.arange(dim // 2, dtype=torch.float32) * (math.log(1000.0) / (dim // 2 - 1)))
    ang = t[:, None] * freqs[None, :]
    return torch.cat([ang.sin(), ang.cos()], dim=1)


def noisy_point(x0: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    """The straight line between data (t = 0) and noise (t = 1)."""
    t = t.view(-1, *([1] * (x0.dim() - 1)))
    return (1 - t) * x0 + t * eps


def flow_target(x0: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    return eps - x0


class VelocityMLP(nn.Module):
    """v(x_t, t, c) for small flattened images. The condition c is an integer id."""

    def __init__(self, dim: int, n_cond: int = 1, hidden: int = 256):
        super().__init__()
        self.cond = nn.Embedding(n_cond, 32)
        self.net = nn.Sequential(
            nn.Linear(dim + 16 + 32, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, xt: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shape = xt.shape
        h = torch.cat([xt.flatten(1), time_features(t), self.cond(c)], dim=1)
        return self.net(h).view(shape)


def flow_loss(model: nn.Module, x0: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    t = torch.rand(x0.shape[0])
    eps = torch.randn_like(x0)
    v = model(noisy_point(x0, t, eps), t, c)
    return ((v - flow_target(x0, eps)) ** 2).mean()


@torch.no_grad()
def sample(model: nn.Module, shape: tuple[int, ...], c: torch.Tensor, steps: int = 50) -> torch.Tensor:
    """Euler integration of the learned velocity from pure noise (t = 1) to data (t = 0)."""
    x = torch.randn(shape)
    dt = 1.0 / steps
    for i in range(steps, 0, -1):
        t = torch.full((shape[0],), i * dt)
        x = x - dt * model(x, t, c)
    return x
