"""Multi-head latent attention (MLA), as in DeepSeek-V2/V3.

Built in the four steps of the tutorial:
1. Compress: c_kv = W_DKV h, one small latent per token (r << H * dh). This is what we cache.
2. Reconstruct: [k_nope; v] = W_UKV c_kv, full per-head keys and values on the fly.
3. Decouple RoPE: a rotation cannot pass through W_UK, so position lives in a
   separate small key k_pe = RoPE(W_KR h), shared by all heads and cached too.
4. Cache r + d_r numbers per token per layer, instead of 2 * Hkv * dh for GQA.

`absorb=True` computes the same output without ever building K or V: W_UK is folded
into the query and W_UV into the output, so attention runs directly on the latents.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .backends import attend
from .config import AttentionConfig
from .heads import merge_heads, split_heads
from .masks import combine, padding_mask, structural_mask
from .rope import RotaryEmbedding, rotate
from .stability import RMSNorm


class MultiHeadLatentAttention(nn.Module):
    """hidden_states (B, N, D) -> (B, N, D).

    Cached per token: c_kv (r) and k_pe (d_r).   Per head: q, k have dn + dr dims, v has dv.
    """

    def __init__(self, config: AttentionConfig, layer_idx: int = 0) -> None:
        super().__init__()
        self.config, self.layer_idx = config, layer_idx
        h, d = config.num_heads, config.hidden_size
        self.dn, self.dr, self.dv, self.r = config.qk_nope_head_dim, config.qk_rope_head_dim, config.v_head_dim, config.kv_lora_rank
        if config.q_lora_rank:  # queries can be low-rank too; this saves activations, not cache
            self.q_down = nn.Linear(d, config.q_lora_rank, bias=False)
            self.q_norm = RMSNorm(config.q_lora_rank)
            self.q_up = nn.Linear(config.q_lora_rank, h * (self.dn + self.dr), bias=False)
        else:
            self.q_proj = nn.Linear(d, h * (self.dn + self.dr), bias=False)
        self.kv_down = nn.Linear(d, self.r, bias=False)  # step 1: W_DKV
        self.kv_norm = RMSNorm(self.r)
        self.kv_up = nn.Linear(self.r, h * (self.dn + self.dv), bias=False)  # step 2: [W_UK; W_UV]
        self.k_rope = nn.Linear(d, self.dr, bias=False)  # step 3: W_KR, one RoPE key for all heads
        self.o_proj = nn.Linear(h * self.dv, d, bias=False)
        self.rotary = RotaryEmbedding(self.dr, config.max_seq_len, config.rope_theta)
        self.scale = 1.0 / math.sqrt(self.dn + self.dr)

    def queries(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x (B, N, D) -> q_nope (B, H, N, dn), q_pe (B, H, N, dr), before RoPE."""
        q = self.q_up(self.q_norm(self.q_down(x))) if self.config.q_lora_rank else self.q_proj(x)
        q = split_heads(q, self.config.num_heads)
        return q.split([self.dn, self.dr], dim=-1)

    def compress(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x (B, N, D) -> c_kv (B, 1, N, r), k_pe (B, 1, N, dr), before RoPE. These get cached."""
        c_kv = self.kv_norm(self.kv_down(x))[:, None]
        k_pe = self.k_rope(x)[:, None]
        return c_kv, k_pe

    def reconstruct(self, c_kv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """c_kv (B, 1, T, r) -> k_nope (B, H, T, dn), v (B, H, T, dv)."""
        kv = split_heads(self.kv_up(c_kv[:, 0]), self.config.num_heads)
        return kv.split([self.dn, self.dv], dim=-1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        kv_cache=None,
        absorb: bool = False,
    ) -> torch.Tensor:
        b, n, _ = hidden_states.shape
        q_nope, q_pe = self.queries(hidden_states)
        c_kv, k_pe = self.compress(hidden_states)

        past = kv_cache.get_seq_length(self.layer_idx) if kv_cache is not None else 0
        if position_ids is None:
            position_ids = torch.arange(past, past + n, device=hidden_states.device)[None].expand(b, n)
        cos, sin = self.rotary(position_ids)
        q_pe, k_pe = rotate(q_pe, cos, sin), rotate(k_pe, cos, sin)  # only the small RoPE parts

        if kv_cache is not None:  # the cache holds latents, not keys and values
            c_kv, k_pe = kv_cache.update(c_kv, k_pe, self.layer_idx)  # (B, 1, T, r), (B, 1, T, dr)

        mask = padding_mask(attention_mask) if attention_mask is not None and attention_mask.dim() == 2 else attention_mask
        if absorb:
            out = self._attend_absorbed(q_nope, q_pe, c_kv, k_pe, mask)
        else:
            k_nope, v = self.reconstruct(c_kv)
            k = torch.cat([k_nope, k_pe.expand(-1, self.config.num_heads, -1, -1)], dim=-1)  # (B, H, T, dn + dr)
            q = torch.cat([q_nope, q_pe], dim=-1)
            out = attend(
                q, k, v, mask, backend=self.config.backend, causal=self.config.causal,
                window_size=self.config.window_size, scale=self.scale,
            )  # (B, H, N, dv)
        return self.o_proj(merge_heads(out))

    def _attend_absorbed(
        self, q_nope: torch.Tensor, q_pe: torch.Tensor, c_kv: torch.Tensor, k_pe: torch.Tensor, mask: torch.Tensor | None
    ) -> torch.Tensor:
        """Attention on latents. q_nope . (W_UK c) = (W_UK^T q_nope) . c, so fold W_UK into q."""
        h = self.config.num_heads
        w = self.kv_up.weight.view(h, self.dn + self.dv, self.r)  # per head: rows of W_UK then W_UV
        w_uk, w_uv = w[:, : self.dn], w[:, self.dn :]  # (H, dn, r), (H, dv, r)

        q_lat = torch.einsum("bhnd,hdr->bhnr", q_nope, w_uk)  # (B, H, N, r): queries in latent space
        scores = (q_lat @ c_kv.transpose(-2, -1) + q_pe @ k_pe.transpose(-2, -1)) * self.scale  # (B, H, N, T)
        allowed = combine(structural_mask(q_nope.size(-2), c_kv.size(-2), self.config.causal, self.config.window_size, q_nope.device), mask)
        if allowed is not None:
            scores = scores.masked_fill(~allowed, float("-inf"))
        weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q_nope.dtype)
        o_lat = weights @ c_kv  # (B, H, N, r): a weighted average of latents
        return torch.einsum("bhnr,hdr->bhnd", o_lat, w_uv)  # (B, H, N, dv): fold W_UV in last
