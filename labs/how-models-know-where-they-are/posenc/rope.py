"""RoPE (Su et al., 2021) in both pairing conventions, plus the quantities the post reasons about.

Half-split ("rotate_half", Llama and Hugging Face): dimension i pairs with i + d/2. Imported from C3.
Interleaved (the RoFormer paper): dimension 2i pairs with 2i + 1.
Both rotate pair i by angle m * theta_i with theta_i = base^(-2i/d).
"""

from __future__ import annotations

import math

import torch

from .c03 import rotate


def inv_freq(head_dim: int, base: float = 10_000.0, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """(d/2,) rotation speeds theta_i in radians per token."""
    return 1.0 / base ** (torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim).to(dtype)


def cos_sin(positions: torch.Tensor, freqs: torch.Tensor, dtype: torch.dtype = torch.float32):
    """positions (N,) -> cos, sin (N, d) in the half-split layout. `dtype` is the dtype the table is built in."""
    angles = positions.to(dtype)[:, None] * freqs.to(dtype)[None]
    emb = torch.cat([angles, angles], dim=-1)
    return emb.cos(), emb.sin()


def apply_half(x: torch.Tensor, positions: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """x (..., N, d) rotated with the half-split convention. Uses C3's `rotate`."""
    cos, sin = cos_sin(positions, freqs)
    shape = x.shape
    x4 = x.reshape(-1, 1, shape[-2], shape[-1])  # (B, 1, N, d) as C3 expects
    return rotate(x4, cos[None], sin[None]).reshape(shape)


def apply_interleaved(x: torch.Tensor, positions: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """x (..., N, d) rotated with the interleaved convention: pairs (x0, x1), (x2, x3), ..."""
    angles = positions.float()[:, None] * freqs.float()[None]  # (N, d/2)
    c, s = angles.cos(), angles.sin()
    x0, x1 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x0 * c - x1 * s, x0 * s + x1 * c], dim=-1).flatten(-2)


def half_to_interleaved(head_dim: int) -> torch.Tensor:
    """Index permutation P with apply_interleaved(x[..., P]) == apply_half(x)[..., P].

    Interleaved slot 2i holds half-split dim i, slot 2i + 1 holds dim i + d/2.
    Converters that move checkpoints between the two layouts apply this to the q and k projection rows.
    """
    half = head_dim // 2
    return torch.stack([torch.arange(half), torch.arange(half) + half], dim=-1).flatten()


def decay_envelope(head_dim: int, base: float, distances: torch.Tensor) -> torch.Tensor:
    """RoFormer's upper bound on |q . R k| from Abel summation, as a function of distance.

    envelope(r) = (1 / (d/2)) * sum_j |S_j(r)|,  S_j(r) = sum_{k<j} exp(i * r * theta_k).
    It bounds a worst case; actual heads can and do exceed what the bound suggests is "typical".
    """
    theta = inv_freq(head_dim, base, torch.float64)
    phase = distances.double()[:, None] * theta[None]
    S = torch.polar(torch.ones_like(phase), phase).cumsum(-1)  # S_1..S_{d/2}
    return S.abs().mean(-1)


def wavelengths(head_dim: int, base: float) -> torch.Tensor:
    """Tokens per full turn of each plane: 2*pi / theta_i."""
    return 2 * math.pi / inv_freq(head_dim, base, torch.float64)


def planes_without_full_turn(head_dim: int, base: float, context: int) -> int:
    """How many planes never complete one rotation inside a context of this length."""
    return int((wavelengths(head_dim, base) > context).sum())


def wrapped_angle_error(positions: torch.Tensor, freqs: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """|angle(dtype) - angle(fp64)| wrapped to [0, pi], per (position, plane)."""
    exact = positions.double()[:, None] * freqs.double()[None]
    approx = (positions.to(dtype)[:, None] * freqs.to(dtype)[None]).double()
    diff = torch.remainder(approx - exact + math.pi, 2 * math.pi) - math.pi
    return diff.abs()
