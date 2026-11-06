"""Grouped-query attention: one module for MHA, MQA and GQA.

    num_kv_heads == num_heads -> MHA  (every query head has its own K and V)
    num_kv_heads == 1         -> MQA  (all query heads share one K and V)
    otherwise                 -> GQA  (groups of n_rep = Hq / Hkv query heads share one)

This is the attention of Llama, Mistral, Qwen and Gemma. It adds RoPE, a KV
cache, optional QK-norm, ALiBi, sliding windows and soft-capping, and hands the
actual softmax(QK^T)V to a pluggable backend.
"""

from __future__ import annotations

import torch
from torch import nn

from .backends import attend
from .biases import alibi_bias, alibi_slopes
from .config import AttentionConfig
from .heads import merge_heads, split_heads
from .masks import padding_mask
from .rope import RotaryEmbedding, apply_rotary
from .stability import RMSNorm


class GroupedQueryAttention(nn.Module):
    """hidden_states (B, N, D) -> (B, N, D).

    Inside:  q (B, Hq, N, dh)   k, v (B, Hkv, N, dh)   cache (B, Hkv, T, dh)
    """

    def __init__(self, config: AttentionConfig, layer_idx: int = 0) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_heads, self.num_kv_heads, self.head_dim = config.num_heads, config.num_kv_heads, config.head_dim
        d, dh = config.hidden_size, config.head_dim
        self.q_proj = nn.Linear(d, self.num_heads * dh, bias=config.qkv_bias)
        self.k_proj = nn.Linear(d, self.num_kv_heads * dh, bias=config.qkv_bias)  # smaller than q_proj for GQA
        self.v_proj = nn.Linear(d, self.num_kv_heads * dh, bias=config.qkv_bias)
        self.o_proj = nn.Linear(self.num_heads * dh, d, bias=False)
        if config.qk_norm:
            self.q_norm, self.k_norm = RMSNorm(dh), RMSNorm(dh)
        if config.positional_encoding == "rope":
            self.rotary = RotaryEmbedding(dh, config.max_seq_len, config.rope_theta)
        if config.positional_encoding == "alibi":
            self.register_buffer("slopes", alibi_slopes(self.num_heads), persistent=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        kv_cache=None,
    ) -> torch.Tensor:
        """attention_mask: (B, Nk) 1/0 padding mask, or a 4D bool/float mask (B|1, 1, Nq, Nk).
        position_ids: (B, N) absolute positions; default continues from the cache length.
        kv_cache: a KVCache / StaticKVCache, updated in place.
        """
        b, n, _ = hidden_states.shape
        q = split_heads(self.q_proj(hidden_states), self.num_heads)  # (B, Hq, N, dh)
        k = split_heads(self.k_proj(hidden_states), self.num_kv_heads)  # (B, Hkv, N, dh)
        v = split_heads(self.v_proj(hidden_states), self.num_kv_heads)  # (B, Hkv, N, dh)

        if self.config.qk_norm:  # normalise before RoPE, as in OLMo 2 and Qwen3
            q, k = self.q_norm(q), self.k_norm(k)

        past = kv_cache.get_seq_length(self.layer_idx) if kv_cache is not None else 0
        if position_ids is None:
            position_ids = torch.arange(past, past + n, device=hidden_states.device)[None].expand(b, n)
        if self.config.positional_encoding == "rope":
            cos, sin = self.rotary(position_ids)
            q, k = apply_rotary(q, k, cos, sin)  # keys are cached already rotated

        if kv_cache is not None:
            k, v = kv_cache.update(k, v, self.layer_idx)  # (B, Hkv, T, dh), T = past + N

        mask = padding_mask(attention_mask) if attention_mask is not None and attention_mask.dim() == 2 else attention_mask
        if self.config.positional_encoding == "alibi":
            bias = alibi_bias(self.slopes, n, k.size(-2)).to(q.dtype)
            if mask is not None and mask.dtype == torch.bool:
                bias = bias.masked_fill(~mask, float("-inf"))
            elif mask is not None:
                bias = bias + mask
            mask = bias

        out = attend(
            q,
            k,
            v,
            mask,
            backend=self.config.backend,
            causal=self.config.causal,
            window_size=self.config.window_size,
            softcap=self.config.softcap,
            dropout_p=self.config.dropout if self.training else 0.0,
        )  # (B, Hq, N, dh)
        return self.o_proj(merge_heads(out))
