"""Grouped-query self-attention with RoPE and an optional KV cache.

This is C3's attnzoo.GroupedQueryAttention, trimmed to what a decoder needs: causal only,
no ALiBi, no windows, no soft-capping. The weight names (q_proj, k_proj, v_proj, o_proj)
match Hugging Face Llama, so checkpoints map one to one.
"""

from __future__ import annotations

import torch
from torch import nn

from .attention_naive import merge_heads, naive_attention, split_heads
from .attention_sdpa import sdpa_attention
from .config import ModelConfig
from .kv_cache import KVCache
from .rope import apply_rotary

BACKENDS = {"naive": naive_attention, "sdpa": sdpa_attention}


class GroupedQueryAttention(nn.Module):
    """hidden_states (B, T, D) -> (B, T, D).

    Inside: q (B, H, T, d)   k, v (B, H_kv, T, d)   cache (B, H_kv, T_past + T, d)
    """

    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.n_rep = config.num_kv_groups
        self.backend = config.attn_backend
        d, hd = config.hidden_size, config.head_dim
        # No biases, as in Llama. K and V are smaller than Q: that is the whole point of GQA.
        self.q_proj = nn.Linear(d, self.num_heads * hd, bias=False)
        self.k_proj = nn.Linear(d, self.num_kv_heads * hd, bias=False)
        self.v_proj = nn.Linear(d, self.num_kv_heads * hd, bias=False)
        self.o_proj = nn.Linear(self.num_heads * hd, d, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: KVCache | None = None,
    ) -> torch.Tensor:
        """cos, sin: (B, T, d), the RoPE tables for these T positions (computed once per model call)."""
        q = split_heads(self.q_proj(hidden_states), self.num_heads)  # (B, H, T, d)
        k = split_heads(self.k_proj(hidden_states), self.num_kv_heads)  # (B, H_kv, T, d)
        v = split_heads(self.v_proj(hidden_states), self.num_kv_heads)  # (B, H_kv, T, d)

        q, k = apply_rotary(q, k, cos, sin)  # keys are cached already rotated
        if kv_cache is not None:
            k, v = kv_cache.update(k, v, self.layer_idx)  # (B, H_kv, T_past + T, d)

        out = BACKENDS[self.backend](q, k, v, self.n_rep)  # (B, H, T, d)
        return self.o_proj(merge_heads(out))
