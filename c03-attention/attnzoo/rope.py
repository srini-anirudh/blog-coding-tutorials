"""Rotary position embeddings (RoPE), applied to queries and keys inside attention.

Convention: the "rotate half" layout used by Llama and Hugging Face, where
dimension i is paired with dimension i + d/2 (not with i + 1).
"""

from __future__ import annotations

import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    """Precomputes cos/sin tables for positions 0..max_seq_len-1.

    inv_freq: (d/2,)   one rotation speed per pair of dimensions
    cos, sin: (max_seq_len, d)
    """

    def __init__(self, head_dim: int, max_seq_len: int = 4096, theta: float = 10_000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)  # (S, d/2): angle = position * speed
        emb = torch.cat([freqs, freqs], dim=-1)  # (S, d): both halves of a pair share an angle
        # Not saved in checkpoints: they are a pure function of the config.
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """position_ids: (B, N) absolute positions -> cos, sin: (B, N, d)."""
        if position_ids.max() >= self.cos_cached.size(0):
            raise ValueError(f"position {int(position_ids.max())} exceeds max_seq_len {self.cos_cached.size(0)}")
        return self.cos_cached[position_ids], self.sin_cached[position_ids]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """(x1, x2) -> (-x2, x1) on the two halves of the last dim: a 90 degree turn of every pair."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate x (B, H, N, d) by its positions. cos/sin: (B, N, d), broadcast over heads."""
    cos, sin = cos[:, None].to(x.dtype), sin[:, None].to(x.dtype)
    return x * cos + rotate_half(x) * sin


def apply_rotary(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate q (B, Hq, N, d) and k (B, Hkv, N, d) with the same tables. Values are never rotated."""
    return rotate(q, cos, sin), rotate(k, cos, sin)
