"""Layers built from `nn.Parameter` and plain tensor ops.

Only `nn.Module` itself is borrowed from PyTorch: it is the bookkeeping
that finds parameters, moves them between devices and saves them.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class Linear(nn.Module):
    """y = x W + b.

    Shapes: x [..., in_features], W [in_features, out_features],
    b [out_features], y [..., out_features].
    `nn.Linear` stores its weight transposed, as [out_features, in_features].
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        # Same scale as nn.Linear: U(-1/sqrt(in), 1/sqrt(in)) keeps the output
        # variance roughly independent of the fan-in.
        bound = 1.0 / math.sqrt(in_features)
        self.weight = nn.Parameter(torch.empty(in_features, out_features).uniform_(-bound, bound))
        self.bias = nn.Parameter(torch.empty(out_features).uniform_(-bound, bound)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x @ self.weight
        if self.bias is not None:
            y = y + self.bias  # [out] broadcasts over every leading dim of x
        return y

    def extra_repr(self) -> str:
        n_in, n_out = self.weight.shape
        return f"in_features={n_in}, out_features={n_out}, bias={self.bias is not None}"


class ReLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.clamp_min(0.0)


class Dropout(nn.Module):
    """Zero each element with probability p in training mode; identity in eval mode."""

    def __init__(self, p: float = 0.1) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        keep = (torch.rand_like(x) >= self.p).to(x.dtype)
        # Scale by 1/(1-p) so the expected activation is the same in both modes.
        return x * keep / (1.0 - self.p)


class Standardize(nn.Module):
    """(x - mean) / std with fixed statistics.

    mean and std are buffers: saved in state_dict and moved by .to(device),
    but not returned by .parameters(), so the optimizer never touches them.
    """

    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("mean", mean.clone())
        self.register_buffer("std", std.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


class MLP(nn.Module):
    """Standardize -> [Linear -> ReLU -> Dropout] x depth -> Linear.

    Shapes: x [B, in_dim] -> logits [B, out_dim].
    """

    def __init__(
        self,
        in_dim: int,
        hidden: int,
        out_dim: int,
        depth: int = 2,
        dropout: float = 0.0,
        mean: torch.Tensor | None = None,
        std: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        mean = torch.zeros(in_dim) if mean is None else mean
        std = torch.ones(in_dim) if std is None else std
        self.norm = Standardize(mean, std)
        dims = [in_dim] + [hidden] * depth
        # ModuleList registers each block, so its parameters are found by .parameters().
        self.blocks = nn.ModuleList(
            nn.Sequential(Linear(d_in, d_out), ReLU(), Dropout(dropout)) for d_in, d_out in zip(dims[:-1], dims[1:])
        )
        self.head = Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        for block in self.blocks:
            h = block(h)
        return self.head(h)


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)
