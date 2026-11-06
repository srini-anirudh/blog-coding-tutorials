"""The two front doors: a row lookup for tokens, a shared linear map for patches.

Text:   ids (B, N) integers   -> E[ids]                     (B, N, d)   E is V x d
Vision: images (B, C, H, W)   -> patchify -> @ W + b        (B, N, d)   W is P*P*C x d, N = (H/P)(W/P)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def onehot_lookup(ids: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """The embedding lookup written as the matmul it really is: onehot(ids) @ E."""
    return F.one_hot(ids, weight.shape[0]).to(weight.dtype) @ weight


def patchify(images: torch.Tensor, patch: int) -> torch.Tensor:
    """(B, C, H, W) -> (B, N, C*P*P), patches in raster order. A reshape: nothing is learned or lost.

    Each patch vector is laid out (channel, row, column), the same order as a flattened Conv2d kernel.
    """
    b, c, h, w = images.shape
    if h % patch or w % patch:
        raise ValueError(f"image {h}x{w} is not divisible by patch {patch}")
    x = images.reshape(b, c, h // patch, patch, w // patch, patch)
    return x.permute(0, 2, 4, 1, 3, 5).reshape(b, (h // patch) * (w // patch), c * patch * patch)


def unpatchify(patches: torch.Tensor, patch: int, channels: int, height: int, width: int) -> torch.Tensor:
    """The exact inverse of `patchify`."""
    b = patches.shape[0]
    gh, gw = height // patch, width // patch
    x = patches.reshape(b, gh, gw, channels, patch, patch).permute(0, 3, 1, 4, 2, 5)
    return x.reshape(b, channels, height, width)


class PatchEmbed(nn.Module):
    """Flatten each P x P patch and apply one shared Linear. Equal to Conv2d(C, d, kernel=P, stride=P)."""

    def __init__(self, patch: int, in_channels: int, dim: int) -> None:
        super().__init__()
        self.patch, self.in_channels = patch, in_channels
        self.proj = nn.Linear(in_channels * patch * patch, dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.proj(patchify(images, self.patch))

    def as_conv(self) -> nn.Conv2d:
        """The same weights as a strided convolution, the way most codebases write it."""
        d = self.proj.out_features
        conv = nn.Conv2d(self.in_channels, d, kernel_size=self.patch, stride=self.patch)
        with torch.no_grad():
            conv.weight.copy_(self.proj.weight.reshape(d, self.in_channels, self.patch, self.patch))
            conv.bias.copy_(self.proj.bias)
        return conv


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())
