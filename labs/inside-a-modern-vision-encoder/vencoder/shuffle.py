"""Pixel shuffle (space-to-depth): fewer tokens, wider channels, nothing thrown away.

The forward op is imported from C7 (`c07-vlm/tinyvlm/projector.py`), which checks it against Idefics3's.
This module adds the inverse and a lossy baseline, 2 x 2 average pooling, to compare against.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

_C07 = Path(__file__).resolve().parents[3] / "c07-vlm"
if str(_C07) not in sys.path:
    sys.path.insert(0, str(_C07))

from tinyvlm.projector import pixel_shuffle  # noqa: E402

__all__ = ["pixel_shuffle", "pixel_unshuffle", "avg_pool_tokens", "unpool_tokens", "patchify", "unpatchify"]


def pixel_unshuffle(tokens: torch.Tensor, factor: int) -> torch.Tensor:
    """Exact inverse of C7's pixel_shuffle: n x (N / f^2) x (d f^2) -> n x N x d."""
    if factor == 1:
        return tokens
    n, m, wide = tokens.shape
    coarse = math.isqrt(m)
    side, dim = coarse * factor, wide // (factor * factor)
    x = tokens.reshape(n, coarse, coarse, wide).permute(0, 2, 1, 3)  # undo the final permute
    x = x.reshape(n, coarse, side, dim * factor).permute(0, 2, 1, 3)  # undo the row merge
    return x.reshape(n, side * side, dim)


def avg_pool_tokens(tokens: torch.Tensor, factor: int) -> torch.Tensor:
    """The lossy alternative: average each f x f block. n x N x d -> n x (N / f^2) x d."""
    n, num, dim = tokens.shape
    side = math.isqrt(num)
    grid = tokens.reshape(n, side, side, dim).permute(0, 3, 1, 2)
    pooled = F.avg_pool2d(grid, factor)
    return pooled.permute(0, 2, 3, 1).reshape(n, -1, dim)


def unpool_tokens(tokens: torch.Tensor, factor: int) -> torch.Tensor:
    """Best guess back from pooled tokens: repeat each one over its f x f block."""
    n, m, dim = tokens.shape
    coarse = math.isqrt(m)
    grid = tokens.reshape(n, coarse, coarse, dim).permute(0, 3, 1, 2)
    up = F.interpolate(grid, scale_factor=factor, mode="nearest")
    return up.permute(0, 2, 3, 1).reshape(n, -1, dim)


def patchify(image: Image.Image, patch: int) -> torch.Tensor:
    """Raw pixel patches as 'features': 1 x (H/p)(W/p) x (p p C). Lets us look at what a token holds."""
    a = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0)
    if a.ndim == 2:
        a = a[..., None]
    h, w, c = a.shape
    x = a.reshape(h // patch, patch, w // patch, patch, c).permute(0, 2, 1, 3, 4)
    return x.reshape(1, (h // patch) * (w // patch), patch * patch * c)


def unpatchify(tokens: torch.Tensor, patch: int, side: int, channels: int = 1) -> np.ndarray:
    x = tokens[0].reshape(side, side, patch, patch, channels).permute(0, 2, 1, 3, 4)
    return x.reshape(side * patch, side * patch, channels).squeeze(-1).numpy()
