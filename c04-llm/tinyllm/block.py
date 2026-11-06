"""One pre-norm decoder block: x + Attn(Norm(x)), then + MLP(Norm(x))."""

from __future__ import annotations

import torch
from torch import nn

from .attention import GroupedQueryAttention
from .config import ModelConfig
from .kv_cache import KVCache
from .mlp import SwiGLU
from .moe import MoE
from .norm import RMSNorm


class DecoderBlock(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = GroupedQueryAttention(config, layer_idx)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        if config.num_experts:
            self.mlp: nn.Module = MoE(
                config.hidden_size, config.intermediate_size, config.num_experts,
                config.num_experts_per_tok, config.expert_capacity_factor,
            )
        else:
            self.mlp = SwiGLU(config.hidden_size, config.intermediate_size)

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, kv_cache: KVCache | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """x: (B, T, D) -> (B, T, D), plus the MoE auxiliary loss (None for a dense block)."""
        # Pre-norm: the residual stream x itself is never normalised, only the branch inputs.
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, kv_cache)
        h = self.mlp(self.post_attention_layernorm(x))
        aux = None
        if isinstance(h, tuple):
            h, aux = h
        return x + h, aux
