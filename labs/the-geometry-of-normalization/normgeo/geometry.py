"""Small geometric helpers shared by the scripts: RoPE, the rotation-angle formula, and a gradient-scaling hook."""

from __future__ import annotations

import math

import torch


def rope(x: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    """Rotate channel pairs (2i, 2i+1) of x (..., T, D) by angle pos * base^(-2i/D)."""
    t, d = x.shape[-2], x.shape[-1]
    theta = torch.arange(t)[:, None] * base ** (-torch.arange(0, d, 2) / d)  # (T, D/2)
    cos, sin = theta.cos(), theta.sin()
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2)


def rotation_angle(h_norm: float, update_norm: float, phi: float) -> float:
    """Angle (radians) between h and h + u, for ||u|| = update_norm at angle phi to h (post section 12)."""
    return math.atan2(update_norm * math.sin(phi), h_norm + update_norm * math.cos(phi))


class GradScale(torch.autograd.Function):
    """Identity in the forward pass. Multiplies the gradient by `scale` in the backward pass."""

    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return g * ctx.scale, None
