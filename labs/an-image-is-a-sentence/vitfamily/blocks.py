"""Blocks. All pre-norm: each sublayer reads LN(x) and adds its output to an untouched residual stream.

ViTBlock        z' = MSA(LN(z)) + z ; z = MLP(LN(z')) + z'
PVTBlock        the same, with spatial-reduction attention
SwinBlock       the same, with C3's WindowAttention (shift = 0 is W-MSA, shift = M // 2 is SW-MSA)
DividedSTBlock  TimeSformer divided attention: time, then space, then MLP. Zero-init temporal projection.
"""

from __future__ import annotations

import torch
from torch import nn

from .attention import MHSA, SRAttention
from .c03 import WindowAttention


class Mlp(nn.Sequential):
    def __init__(self, dim: int, ratio: int = 4) -> None:
        super().__init__(nn.Linear(dim, ratio * dim), nn.GELU(), nn.Linear(ratio * dim, dim))


class ViTBlock(nn.Module):
    def __init__(self, dim: int, heads: int, attn: nn.Module | None = None) -> None:
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = attn if attn is not None else MHSA(dim, heads)
        self.mlp = Mlp(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class PVTBlock(ViTBlock):
    def __init__(self, dim: int, heads: int, ratio: int) -> None:
        super().__init__(dim, heads, SRAttention(dim, heads, ratio))

    def forward(self, x: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), hw)
        return x + self.mlp(self.norm2(x))


class SwinBlock(nn.Module):
    """x (B, H, W, C) -> (B, H, W, C). Window attention, relative bias, cyclic shift and mask are C3's."""

    def __init__(self, dim: int, heads: int, window: int = 7, shift: int = 0) -> None:
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, heads, window, shift)
        self.mlp = Mlp(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class DividedSTBlock(nn.Module):
    """x (B, T, N, D). Temporal attention over the T copies of each patch, then spatial attention per frame.

    zero_init_temporal=True sets the temporal output projection to zero (ViViT's initialisation), so a
    fresh block computes exactly what the image block it was copied from computes, frame by frame.
    """

    def __init__(self, dim: int, heads: int, zero_init_temporal: bool = True) -> None:
        super().__init__()
        self.norm_t, self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn_t = MHSA(dim, heads)  # its own W_Q, W_K, W_V, W_O: 2x the attention parameters
        self.attn = MHSA(dim, heads)
        self.mlp = Mlp(dim)
        if zero_init_temporal:
            nn.init.zeros_(self.attn_t.proj.weight)
            nn.init.zeros_(self.attn_t.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, n, d = x.shape
        xt = x.transpose(1, 2).reshape(b * n, t, d)  # one sequence of length T per patch
        xt = xt + self.attn_t(self.norm_t(xt))
        x = xt.view(b, n, t, d).transpose(1, 2).reshape(b * t, n, d)  # one sequence of length N per frame
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x.view(b, t, n, d)

    def spatial_only(self, x: torch.Tensor) -> torch.Tensor:
        """The image block this one started from: no temporal sublayer."""
        b, t, n, d = x.shape
        x = x.reshape(b * t, n, d)
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x.view(b, t, n, d)
