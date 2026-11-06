"""Activation checkpointing on C4's TinyLLM: none, full (per block) and selective.

Full: each block keeps only its input and reruns its own forward during backward.
Selective: the GEMM outputs are kept, and only the cheap elementwise ops (the two RMSNorms and the
SiLU-multiply) are recomputed. Those are cheap in FLOPs and expensive in memory.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint


class FullCheckpoint(nn.Module):
    def __init__(self, block: nn.Module) -> None:
        super().__init__()
        self.block = block

    def forward(self, x, cos, sin, kv_cache=None):
        if not torch.is_grad_enabled():
            return self.block(x, cos, sin, kv_cache)
        return checkpoint(self.block, x, cos, sin, kv_cache, use_reentrant=False)


class RecomputedNorm(nn.Module):
    """RMSNorm whose internals (an FP32 upcast, the inverse RMS, the scaled copy) are not saved."""

    def __init__(self, norm: nn.Module) -> None:
        super().__init__()
        self.norm = norm

    def forward(self, x):
        return checkpoint(self.norm, x, use_reentrant=False) if torch.is_grad_enabled() else self.norm(x)


def _silu_mul(g, u):
    return F.silu(g) * u


class SelectiveSwiGLU(nn.Module):
    """down(silu(gate(x)) * up(x)) with the SiLU-multiply recomputed: silu(gate(x)) is never saved."""

    def __init__(self, mlp: nn.Module) -> None:
        super().__init__()
        self.mlp = mlp

    def forward(self, x):
        g, u = self.mlp.gate_proj(x), self.mlp.up_proj(x)
        h = checkpoint(_silu_mul, g, u, use_reentrant=False) if torch.is_grad_enabled() else _silu_mul(g, u)
        return self.mlp.down_proj(h)


def apply(model: nn.Module, mode: str) -> nn.Module:
    """mode is 'none', 'full' or 'selective'. Changes the model in place and returns it."""
    if mode == "none":
        return model
    for i, block in enumerate(model.layers):
        if mode == "full":
            model.layers[i] = FullCheckpoint(block)
        elif mode == "selective":
            block.input_layernorm = RecomputedNorm(block.input_layernorm)
            block.post_attention_layernorm = RecomputedNorm(block.post_attention_layernorm)
            block.mlp = SelectiveSwiGLU(block.mlp)
        else:
            raise ValueError(mode)
    return model
