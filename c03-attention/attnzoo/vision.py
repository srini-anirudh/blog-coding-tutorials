"""Attention over image patches: global (ViT) and windowed (Swin).

A ViT layer is the same MHA as a text encoder, with no causal mask:
patches have no order in which they "arrive".
"""

from __future__ import annotations

import torch
from torch import nn

from .config import AttentionConfig
from .gqa import GroupedQueryAttention
from .pos2d import RelativePositionBias2D


def patchify(images: torch.Tensor, patch: int) -> torch.Tensor:
    """(B, C, H, W) -> (B, N, C*p*p) with N = (H/p) * (W/p), patches in row-major order."""
    b, c, h, w = images.shape
    x = images.unfold(2, patch, patch).unfold(3, patch, patch)  # (B, C, H/p, W/p, p, p)
    return x.permute(0, 2, 3, 1, 4, 5).reshape(b, (h // patch) * (w // patch), c * patch * patch)


def vit_attention_config(hidden_size: int, num_heads: int, backend: str = "sdpa") -> AttentionConfig:
    """The ViT layer: bidirectional MHA, positions added to the patch embeddings beforehand."""
    return AttentionConfig("mha", hidden_size, num_heads, causal=False, positional_encoding="none", backend=backend)


def window_partition(x: torch.Tensor, window: int) -> torch.Tensor:
    """(B, H, W, C) -> (B * nW, w*w, C): each window becomes its own short sequence."""
    b, h, w, c = x.shape
    x = x.view(b, h // window, window, w // window, window, c)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(-1, window * window, c)


def window_merge(windows: torch.Tensor, window: int, height: int, width: int) -> torch.Tensor:
    """(B * nW, w*w, C) -> (B, H, W, C), the exact inverse of window_partition."""
    c = windows.size(-1)
    x = windows.view(-1, height // window, width // window, window, window, c)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(-1, height, width, c)


def shifted_window_mask(height: int, width: int, window: int, shift: int) -> torch.Tensor:
    """(nW, w*w, w*w) bool. After a cyclic shift, a window can hold pixels from opposite
    image borders; label the regions and only let a token attend inside its own region."""
    region = torch.zeros(height, width, dtype=torch.long)
    cuts = (slice(0, -window), slice(-window, -shift), slice(-shift, None))
    label = 0
    for hs in cuts:
        for ws in cuts:
            region[hs, ws] = label
            label += 1
    ids = window_partition(region[None, :, :, None], window).squeeze(-1)  # (nW, w*w)
    return ids[:, :, None] == ids[:, None, :]


class WindowAttention(nn.Module):
    """Self-attention inside non-overlapping w x w windows. x (B, H, W, C) -> (B, H, W, C).

    Cost per layer is O(H W w^2) instead of O((H W)^2). shift = w // 2 on every other
    layer lets information cross window borders (Swin, Liu et al., 2021).
    """

    def __init__(self, hidden_size: int, num_heads: int, window: int, shift: int = 0, backend: str = "sdpa") -> None:
        super().__init__()
        self.window, self.shift = window, shift
        self.attn = GroupedQueryAttention(vit_attention_config(hidden_size, num_heads, backend))
        self.rel_bias = RelativePositionBias2D(window, num_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, h, w, _ = x.shape
        if self.shift:
            x = torch.roll(x, shifts=(-self.shift, -self.shift), dims=(1, 2))
        windows = window_partition(x, self.window)  # (B*nW, w*w, C)
        bias = self.rel_bias().to(x.dtype)  # (1, heads, w*w, w*w)
        if self.shift:
            mask = shifted_window_mask(h, w, self.window, self.shift).to(x.device)  # (nW, w*w, w*w)
            mask = mask.repeat(b, 1, 1)[:, None]  # (B*nW, 1, w*w, w*w); windows are batch-major
            bias = bias.masked_fill(~mask, float("-inf"))
        out = self.attn(windows, attention_mask=bias)
        out = window_merge(out, self.window, h, w)
        if self.shift:
            out = torch.roll(out, shifts=(self.shift, self.shift), dims=(1, 2))
        return out
