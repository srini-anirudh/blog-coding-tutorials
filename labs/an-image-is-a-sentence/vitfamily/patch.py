"""Tokenisers: the only layers in a ViT that a language model does not have.

PatchEmbed           one Conv2d with kernel == stride == P: patch-and-project
patch_embed_matmul   the same thing written as reshape + one matrix multiply
merge_patches        Swin's patch merging: concatenate each 2x2 group, project 4C -> 2C
TubeletEmbed         the video version: a Conv3d over t x P x P volumes
resize_pos_table     bicubic resize of a learned position table (224 -> 384), with the CLS-row bug on demand
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .c03 import patchify


class PatchEmbed(nn.Module):
    """(B, C, H, W) -> (B, N, D), N = HW / P^2. Non-overlapping because kernel == stride."""

    def __init__(self, patch: int, dim: int, in_chans: int = 3) -> None:
        super().__init__()
        self.patch = patch
        self.proj = nn.Conv2d(in_chans, dim, kernel_size=patch, stride=patch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)  # row-major token order


def patch_embed_matmul(images: torch.Tensor, conv: nn.Conv2d) -> torch.Tensor:
    """Flatten each patch to P*P*C numbers and multiply by E (P*P*C, D). Same numbers as `conv`."""
    p = conv.kernel_size[0]
    e = conv.weight.flatten(1).T  # (C*P*P, D); C-major flattening, as patchify produces
    return patchify(images, p) @ e + conv.bias


def merge_patches(x: torch.Tensor) -> torch.Tensor:
    """(B, H, W, C) -> (B, H/2, W/2, 4C), in Swin's order: (0,0), (1,0), (0,1), (1,1)."""
    return torch.cat([x[:, 0::2, 0::2], x[:, 1::2, 0::2], x[:, 0::2, 1::2], x[:, 1::2, 1::2]], dim=-1)


def pixel_unshuffle_order(channels: int) -> torch.Tensor:
    """Index that reorders merge_patches' 4C channels into F.pixel_unshuffle's (c * 4 + dy * 2 + dx)."""
    swin_slot = {(0, 0): 0, (1, 0): 1, (0, 1): 2, (1, 1): 3}
    order = [swin_slot[(dy, dx)] * channels + c for c in range(channels) for dy in (0, 1) for dx in (0, 1)]
    return torch.tensor(order)


class TubeletEmbed(nn.Module):
    """(B, C, T, H, W) -> (B, (T/t) * (H/P) * (W/P), D). t = 1 is uniform frame sampling."""

    def __init__(self, t: int, patch: int, dim: int, in_chans: int = 3) -> None:
        super().__init__()
        self.proj = nn.Conv3d(in_chans, dim, kernel_size=(t, patch, patch), stride=(t, patch, patch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)


def resize_pos_table(table: torch.Tensor, new_side: int, n_special: int = 1, hold_out: bool = True) -> torch.Tensor:
    """table (1, n_special + s*s, D) -> (1, n_special + new_side^2, D), bicubic on the s x s grid.

    n_special is 1 for ViT (CLS) and 2 for distilled DeiT (CLS, distillation). hold_out=False is the
    silent bug: the special rows are treated as the first patches, so every patch position is read
    n_special slots late, and the last patch rows end up as the special-token embeddings.
    """
    n = table.size(1) - n_special
    side = int(round(n**0.5))
    if side * side != n:
        raise ValueError(f"{n} patch positions is not a square grid")
    if hold_out:
        special, grid = table[:, :n_special], table[:, n_special:]
    else:
        grid, special = table[:, :n], table[:, n:]
    d = table.size(-1)
    grid = grid.reshape(1, side, side, d).permute(0, 3, 1, 2)
    grid = F.interpolate(grid, size=(new_side, new_side), mode="bicubic", align_corners=False)
    grid = grid.permute(0, 2, 3, 1).reshape(1, new_side * new_side, d)
    return torch.cat([special, grid], dim=1)
