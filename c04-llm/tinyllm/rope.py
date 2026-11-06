"""Rotary position embedding (RoPE), in the "rotate half" layout used by Llama and Hugging Face.

Same design and names as C3's attnzoo.rope: a cos/sin table indexed by absolute position,
applied to queries and keys after projection and before the dot product. Values are never rotated.
"""

from __future__ import annotations

import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    """Precomputes cos/sin tables for positions 0 .. max_seq_len - 1.

    inv_freq: (d/2,)   one rotation speed per pair of channels
    cos, sin: (max_seq_len, d)
    """

    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10_000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        # Angles in float32: position * speed gets large, and bf16 would round it badly.
        freqs = torch.outer(positions, inv_freq)  # (S, d/2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (S, d): channel i and channel i + d/2 share an angle
        # Not persistent: a pure function of the config, so it does not belong in a checkpoint.
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """position_ids: (B, T) absolute positions -> cos, sin: (B, T, d), float32."""
        return self.cos_cached[position_ids], self.sin_cached[position_ids]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """(x1, x2) -> (-x2, x1) on the two halves of the last dim: a 90 degree turn of every pair.

    Channel i is paired with channel i + d/2. This pairing is a convention. It must match
    the checkpoint, or trained weights read positions wrongly (see "Break it" in the post).
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate x (B, H, T, d) by its positions. cos, sin: (B, T, d), broadcast over heads."""
    cos, sin = cos[:, None].to(x.dtype), sin[:, None].to(x.dtype)
    return x * cos + rotate_half(x) * sin


def apply_rotary(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate q (B, H, T, d) and k (B, H_kv, T, d) with the same tables."""
    return rotate(q, cos, sin), rotate(k, cos, sin)
