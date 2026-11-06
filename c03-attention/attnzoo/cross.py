"""Cross-attention: queries from one sequence, keys and values from another.

    Q = X_q W_Q        K = X_s W_K        V = X_s W_V
The output has the query sequence's length; the context can be any length and width.
Text -> text (encoder-decoder), text -> image (Flamingo, Llama 3.2 Vision), and
latent queries -> image (Q-Former, Perceiver) are all this one module.
"""

from __future__ import annotations

import torch
from torch import nn

from .backends import attend
from .config import AttentionConfig
from .heads import merge_heads, split_heads
from .masks import padding_mask


class CrossAttention(nn.Module):
    """hidden_states (B, Nq, D), context (B, Ns, Dc) -> (B, Nq, D).

    No causal mask (every query may read the whole context) and no RoPE (the two
    sequences have unrelated positions). context_mask: (B, Ns) 1 = real, 0 = pad.
    """

    def __init__(self, config: AttentionConfig) -> None:
        super().__init__()
        self.config = config
        h, hkv, dh = config.num_heads, config.num_kv_heads, config.head_dim
        self.q_proj = nn.Linear(config.hidden_size, h * dh, bias=config.qkv_bias)
        self.k_proj = nn.Linear(config.context_size, hkv * dh, bias=config.qkv_bias)  # from the context width
        self.v_proj = nn.Linear(config.context_size, hkv * dh, bias=config.qkv_bias)
        self.o_proj = nn.Linear(h * dh, config.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor, context: torch.Tensor, context_mask: torch.Tensor | None = None) -> torch.Tensor:
        q = split_heads(self.q_proj(hidden_states), self.config.num_heads)  # (B, H, Nq, dh)
        k = split_heads(self.k_proj(context), self.config.num_kv_heads)  # (B, Hkv, Ns, dh)
        v = split_heads(self.v_proj(context), self.config.num_kv_heads)
        mask = padding_mask(context_mask) if context_mask is not None else None
        out = attend(q, k, v, mask, backend=self.config.backend, causal=False)  # (B, H, Nq, dh)
        return self.o_proj(merge_heads(out))


class FeedForward(nn.Module):
    """A plain GELU MLP, used by the cross-attention blocks and resamplers."""

    def __init__(self, dim: int, mult: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, mult * dim), nn.GELU(), nn.Linear(mult * dim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GatedCrossAttentionBlock(nn.Module):
    """Flamingo-style block inserted between frozen LM layers (Alayrac et al., 2022).

    x <- x + tanh(a_attn) * CrossAttn(LN(x), image)
    x <- x + tanh(a_ff)   * FFN(x)
    Both gates start at 0, so at initialisation the block is the identity and the
    frozen language model behaves exactly as before. Training opens the gates.
    """

    def __init__(self, config: AttentionConfig) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(config.hidden_size)
        self.cross = CrossAttention(config)
        self.ffn = FeedForward(config.hidden_size)
        self.attn_gate = nn.Parameter(torch.zeros(1))
        self.ff_gate = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, image_tokens: torch.Tensor, image_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + torch.tanh(self.attn_gate) * self.cross(self.norm(x), image_tokens, image_mask)
        return x + torch.tanh(self.ff_gate) * self.ffn(x)
