"""The classic forms, written separately so each idea is visible on its own.

MultiHeadAttention   three projections, H heads of K and V
FusedQKVAttention    the same maths with one wide projection, split afterwards
MultiQueryAttention  H query heads, a single shared K and V head

GroupedQueryAttention (gqa.py) unifies all three; the tests check it against these.
"""

from __future__ import annotations

import torch
from torch import nn

from .backends import attend
from .core import scaled_dot_product_attention
from .heads import merge_heads, split_heads
from .masks import causal_mask


class MultiHeadAttention(nn.Module):
    """x (B, N, D) -> (B, N, D), with D = num_heads * head_dim."""

    def __init__(self, hidden_size: int, num_heads: int, causal: bool = False, backend: str = "naive") -> None:
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.num_heads, self.causal, self.backend = num_heads, causal, backend
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        q = split_heads(self.q_proj(x), self.num_heads)  # (B, H, N, dh)
        k = split_heads(self.k_proj(x), self.num_heads)
        v = split_heads(self.v_proj(x), self.num_heads)
        out = attend(q, k, v, attn_mask, backend=self.backend, causal=self.causal)  # (B, H, N, dh)
        return self.o_proj(merge_heads(out))  # mix information across heads


class FusedQKVAttention(nn.Module):
    """One (D -> 3D) matmul instead of three (D -> D) ones: fewer kernel launches, same result."""

    def __init__(self, hidden_size: int, num_heads: int, causal: bool = False, backend: str = "naive") -> None:
        super().__init__()
        self.num_heads, self.causal, self.backend = num_heads, causal, backend
        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)  # three (B, N, D) views of one (B, N, 3D) tensor
        q, k, v = (split_heads(t, self.num_heads) for t in (q, k, v))
        out = attend(q, k, v, attn_mask, backend=self.backend, causal=self.causal)
        return self.o_proj(merge_heads(out))

    @classmethod
    def from_separate(cls, mha: MultiHeadAttention) -> "FusedQKVAttention":
        """Stack W_q, W_k, W_v row-wise into one weight: the fused layer computes the same thing."""
        hidden = mha.q_proj.in_features
        fused = cls(hidden, mha.num_heads, mha.causal, mha.backend)
        with torch.no_grad():
            fused.qkv_proj.weight.copy_(torch.cat([mha.q_proj.weight, mha.k_proj.weight, mha.v_proj.weight], dim=0))
            fused.o_proj.weight.copy_(mha.o_proj.weight)
        return fused


class MultiQueryAttention(nn.Module):
    """H query heads, one K head and one V head (Shazeer, 2019). The cache shrinks H times."""

    def __init__(self, hidden_size: int, num_heads: int, causal: bool = True) -> None:
        super().__init__()
        self.num_heads, self.causal = num_heads, causal
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.head_dim, bias=False)  # D -> dh, not D -> D
        self.v_proj = nn.Linear(hidden_size, self.head_dim, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = split_heads(self.q_proj(x), self.num_heads)  # (B, H, N, dh)
        k = split_heads(self.k_proj(x), 1)  # (B, 1, N, dh)
        v = split_heads(self.v_proj(x), 1)
        # No copy needed: the size-1 head dim of k broadcasts against H in the matmul.
        mask = causal_mask(x.size(1), device=x.device) if self.causal else None
        out, _ = scaled_dot_product_attention(q, k, v, mask=mask)  # (B, H, N, dh)
        return self.o_proj(merge_heads(out))
