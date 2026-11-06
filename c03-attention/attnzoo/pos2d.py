"""Positions on a 2D grid of image patches.

sincos_2d_embedding     absolute, added to patch embeddings (MAE, SigLIP-style)
RelativePositionBias2D  a learned bias per (dy, dx) inside a window (Swin)
Rope2D                  rotate half the head dims by row and half by column (axial 2D RoPE)
"""

from __future__ import annotations

import torch
from torch import nn

from .rope import RotaryEmbedding, rotate


def sincos_1d(positions: torch.Tensor, dim: int) -> torch.Tensor:
    """(N,) positions -> (N, dim) sin/cos features at geometric frequencies."""
    omega = 1.0 / 10_000 ** (torch.arange(dim // 2, dtype=torch.float32) / (dim // 2))
    angles = positions.float()[:, None] * omega[None]
    return torch.cat([angles.sin(), angles.cos()], dim=-1)


def sincos_2d_embedding(height: int, width: int, dim: int) -> torch.Tensor:
    """(H*W, dim): half the channels encode the row, half the column."""
    rows, cols = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    return torch.cat([sincos_1d(rows.flatten(), dim // 2), sincos_1d(cols.flatten(), dim // 2)], dim=-1)


class RelativePositionBias2D(nn.Module):
    """Swin's bias: one learned scalar per head for each offset (dy, dx) in a w x w window.

    table: ((2w-1)^2, H). index: (w*w, w*w) maps each query-key pair to its table row.
    """

    def __init__(self, window: int, num_heads: int) -> None:
        super().__init__()
        self.table = nn.Parameter(torch.zeros((2 * window - 1) ** 2, num_heads))
        coords = torch.stack(torch.meshgrid(torch.arange(window), torch.arange(window), indexing="ij")).flatten(1)  # (2, w*w)
        rel = coords[:, :, None] - coords[:, None, :] + (window - 1)  # (2, w*w, w*w), shifted to >= 0
        self.register_buffer("index", rel[0] * (2 * window - 1) + rel[1], persistent=False)

    def forward(self) -> torch.Tensor:
        """Returns (1, H, w*w, w*w), ready to add to the logits."""
        return self.table[self.index].permute(2, 0, 1)[None]


class Rope2D(nn.Module):
    """Axial 2D RoPE: the score between two patches depends only on (dy, dx).

    The first half of each head is rotated by the row index, the second by the column.
    """

    def __init__(self, head_dim: int, max_side: int = 64, theta: float = 10_000.0) -> None:
        super().__init__()
        if head_dim % 4:
            raise ValueError("2D RoPE splits head_dim in two halves of rotated pairs: need head_dim % 4 == 0")
        self.rope = RotaryEmbedding(head_dim // 2, max_side, theta)

    def forward(self, x: torch.Tensor, rows: torch.Tensor, cols: torch.Tensor) -> torch.Tensor:
        """x (B, H, N, d), rows/cols (N,) integer grid coordinates -> rotated x."""
        x_row, x_col = x.chunk(2, dim=-1)
        cos_r, sin_r = self.rope(rows[None])
        cos_c, sin_c = self.rope(cols[None])
        return torch.cat([rotate(x_row, cos_r, sin_r), rotate(x_col, cos_c, sin_c)], dim=-1)
