"""Part VII of the post: Adam written out so its state can be inspected, plus clipping and a gradient stream."""

from __future__ import annotations

import math

import torch


class InspectableAdam:
    """Adam on one flat tensor, step by step, exposing m_hat, sqrt(v_hat) and the update.

    Same arithmetic as torch.optim.Adam (bias-corrected moments, eps added after the square root).
    """

    def __init__(self, n: int, lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8,
                 dtype: torch.dtype = torch.float64) -> None:
        self.lr, (self.b1, self.b2), self.eps = lr, betas, eps
        self.m = torch.zeros(n, dtype=dtype)
        self.v = torch.zeros(n, dtype=dtype)
        self.t = 0

    def step(self, grad: torch.Tensor, lr: float | None = None) -> dict:
        lr = self.lr if lr is None else lr
        self.t += 1
        self.m.mul_(self.b1).add_(grad, alpha=1 - self.b1)
        self.v.mul_(self.b2).addcmul_(grad, grad, value=1 - self.b2)
        m_hat = self.m / (1 - self.b1**self.t)
        v_hat = self.v / (1 - self.b2**self.t)
        update = -lr * m_hat / (v_hat.sqrt() + self.eps)
        return {"m_hat": m_hat, "sqrt_v_hat": v_hat.sqrt(), "update": update}


def stress_stream(steps: int = 2000, n: int = 256, seed: int = 0) -> torch.Tensor:
    """The synthetic gradient stream of the post's Figure 10, one row per step.

        steps 1-20      hot start: noise with std 10
        steps 21-399    ordinary: noise with std 1
        steps 400-1499  quiet stretch: noise with std 0.01
        steps 1500-1559 burst: correlated gradients, mean 1 with noise std 0.3 (same sign every step)
        steps 1560-     quiet again
    """
    g = torch.Generator().manual_seed(seed)
    noise = torch.randn(steps, n, generator=g, dtype=torch.float64)
    scale = torch.ones(steps, 1, dtype=torch.float64)
    scale[:20] = 10.0
    scale[399:] = 0.01
    grads = noise * scale
    grads[1499:1559] = 1.0 + 0.3 * noise[1499:1559]
    return grads


def clip_global_norm(grads: list[torch.Tensor], tau: float) -> tuple[list[torch.Tensor], float]:
    """Global-norm clipping: one norm over every tensor, one scale factor, direction unchanged."""
    total = math.sqrt(sum(g.pow(2).sum().item() for g in grads))
    scale = min(1.0, tau / (total + 1e-12))
    return [g * scale for g in grads], total
