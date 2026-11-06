"""Normalization operators written from scratch, plus their hand-derived backward passes.

The tensor convention is x[n, c, s]: sample n, feature or channel c, position s (post section 02).
Every operator below is the same arithmetic pointed at different axes.
"""

from __future__ import annotations

import torch
from torch import nn

# Which axes each operator averages over, for x of shape (N, C, S).
AXES = {
    "batchnorm": (0, 2),  # one statistic per channel, shared by the whole batch
    "layernorm": (1,),  # one statistic per (sample, position): per token
    "instancenorm": (2,),  # one statistic per (sample, channel)
    "rmsnorm": (1,),  # LayerNorm's axes, without the mean
}


def normalize(x: torch.Tensor, dims: tuple[int, ...], center: bool = True, eps: float = 1e-5) -> torch.Tensor:
    """(x - mean) / sqrt(var + eps) over `dims`. With center=False this is x / rms(x)."""
    if center:
        x = x - x.mean(dims, keepdim=True)
    return x * torch.rsqrt(x.pow(2).mean(dims, keepdim=True) + eps)


def atlas(x: torch.Tensor, kind: str, groups: int = 2, eps: float = 1e-5) -> torch.Tensor:
    """Apply one named operator (no learned gain or bias) to x of shape (N, C, S)."""
    if kind == "groupnorm":  # split C into groups, then average over (channels in group, positions)
        n, c, s = x.shape
        return normalize(x.reshape(n, groups, c // groups, s), (2, 3), eps=eps).reshape(n, c, s)
    return normalize(x, AXES[kind], center=kind != "rmsnorm", eps=eps)


# ---- Backward passes, derived by hand (post section 03) -------------------------------------------


def layernorm_backward(x: torch.Tensor, gamma: torch.Tensor, dy: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """dL/dx for y = gamma * (x - mu) / sigma over the last dim.

    dx = (1/sigma) * [g - mean(g) - xhat * mean(g * xhat)],  with g = gamma * dy.
    The bracket removes the component of g along the all-ones vector and along xhat.
    """
    mu = x.mean(-1, keepdim=True)
    sigma = torch.sqrt((x - mu).pow(2).mean(-1, keepdim=True) + eps)
    xhat = (x - mu) / sigma
    g = gamma * dy
    return (g - g.mean(-1, keepdim=True) - xhat * (g * xhat).mean(-1, keepdim=True)) / sigma


def rmsnorm_backward(x: torch.Tensor, gamma: torch.Tensor, dy: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """dL/dx for y = gamma * x / rms(x). Same as LayerNorm's bracket without the -mean(g) term."""
    r = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    xhat = x / r
    g = gamma * dy
    return (g - xhat * (g * xhat).mean(-1, keepdim=True)) / r


# ---- Modules ----------------------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """y = w * x / sqrt(mean(x^2) + eps) over the last dim.

    Options that matter when porting checkpoints (post section 11):
      eps_inside=False  ->  x / (sqrt(mean(x^2)) + eps), the T5 / some-Flax convention.
      offset=True       ->  gain is (1 + w) with w initialised at 0, the Gemma convention.
      upcast=False      ->  do the reduction in the input dtype (what you should not do in bf16).
    """

    def __init__(self, dim: int, eps: float = 1e-6, eps_inside: bool = True, offset: bool = False, upcast: bool = True):
        super().__init__()
        self.eps, self.eps_inside, self.offset, self.upcast = eps, eps_inside, offset, upcast
        self.weight = nn.Parameter(torch.zeros(dim) if offset else torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        h = x.float() if self.upcast else x
        ms = h.pow(2).mean(-1, keepdim=True)
        h = h * torch.rsqrt(ms + self.eps) if self.eps_inside else h / (ms.sqrt() + self.eps)
        gain = 1.0 + self.weight if self.offset else self.weight
        return (gain.to(h.dtype) * h).to(dtype)


class DyT(nn.Module):
    """Dynamic Tanh (Zhu et al., 2025): gamma * tanh(alpha * x) + beta. No statistics are computed."""

    def __init__(self, dim: int, alpha0: float = 0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha0))
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * torch.tanh(self.alpha * x) + self.bias


def make_norm(kind: str, dim: int) -> nn.Module:
    if kind == "rms":
        return RMSNorm(dim)
    if kind == "ln":
        return nn.LayerNorm(dim)
    if kind == "dyt":
        return DyT(dim)
    raise ValueError(kind)


def weight_decay_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """Two optimizer groups: matrices get decay. Norm gains, biases and embeddings do not (post section 11, item 3)."""
    decay, no_decay = [], []
    for module_name, module in model.named_modules():
        for name, p in module.named_parameters(recurse=False):
            skip = p.ndim < 2 or isinstance(module, nn.Embedding)
            (no_decay if skip else decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
