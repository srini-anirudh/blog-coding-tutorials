"""Position on a patch grid: raster ids, and resizing a learned table for a new resolution.

Axial 2D RoPE and the 2D sin-cos table come from C3 (`attnzoo.pos2d`), re-exported by `posenc.c03`.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F


def grid_coords(h: int, w: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Row and column of each patch in raster order. Both (h*w,)."""
    rows, cols = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    return rows.flatten(), cols.flatten()


def raster_id(row: int, col: int, w: int) -> int:
    """The 1D position a flattened grid gives a patch."""
    return row * w + col


def resize_pos_table(table: torch.Tensor, old: int, new: int, has_cls: bool = True, antialias: bool = False):
    """Resize an (optional cls row +) old*old x d table to new*new, the way ViT, DINOv2 and SigLIP fine-tune.

    The cls row is not a grid cell: it is split off, kept as is, and put back in front.
    """
    cls, grid = (table[:1], table[1:]) if has_cls else (table[:0], table)
    if grid.shape[0] != old * old:
        raise ValueError(f"{grid.shape[0]} grid rows is not {old}x{old}; did you forget the cls row?")
    d = grid.shape[1]
    img = grid.reshape(1, old, old, d).permute(0, 3, 1, 2)
    out = F.interpolate(img, size=(new, new), mode="bicubic", align_corners=False, antialias=antialias)
    return torch.cat([cls, out.permute(0, 2, 3, 1).reshape(new * new, d)], 0)
