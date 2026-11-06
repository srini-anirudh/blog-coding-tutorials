"""LayerNorm and RMSNorm written as their reductions, and what QK-norm does to attention logits."""

from __future__ import annotations

import math

import torch

from . import _paths  # noqa: F401
from tinyllm.norm import RMSNorm


def layer_norm_two_reductions(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """LayerNorm without gain or shift: reduction 1 is the mean, reduction 2 the variance about it."""
    mu = x.mean(-1, keepdim=True)  # reduction 1
    xc = x - mu
    var = xc.pow(2).mean(-1, keepdim=True)  # reduction 2, which needs reduction 1 first
    return xc / torch.sqrt(var + eps)


def rms_norm_one_reduction(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """RMSNorm without gain: one reduction, the mean square."""
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def attention_entropy(p: torch.Tensor) -> torch.Tensor:
    """Mean entropy in nats of attention rows p (..., T). log(T) is uniform, 0 is one-hot."""
    return -(p * p.clamp_min(1e-30).log()).sum(-1).mean()


def qk_logits(x: torch.Tensor, wq: torch.Tensor, wk: torch.Tensor, qk_norm: bool) -> torch.Tensor:
    """Causal attention logits q.k / sqrt(d_h) for one head, with or without RMSNorm on q and k."""
    q, k = x @ wq, x @ wk
    if qk_norm:
        norm = RMSNorm(q.shape[-1])  # gain 1, as at initialization
        q, k = norm(q), norm(k)
    s = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
    t = x.shape[-2]
    return s.masked_fill(~torch.ones(t, t, dtype=torch.bool).tril(), float("-inf"))


def softmax_grad_norm(logits: torch.Tensor) -> float:
    """||d sum(softmax(z) * u) / dz|| for a fixed random u: how much gradient passes back through softmax."""
    z = logits.detach().clone().requires_grad_(True)
    u = torch.randn(z.shape, generator=torch.Generator().manual_seed(0), dtype=z.dtype)
    (z.softmax(-1) * u).sum().backward()
    g = z.grad.nan_to_num(0.0)
    return g.norm().item()
