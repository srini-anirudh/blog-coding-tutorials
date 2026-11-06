"""The identical middle: one attention + MLP block that both models use, line for line.

Nothing in this file knows whether its input came from words or from pixels. The only modality-shaped input is
`mask`, which the caller passes in.

BlockConfig switches between the two families' accessories:
  ViT style   norm="layer", mlp="gelu",   bias=True,  layerscale=1e-4 (optional)
  LLM style   norm="rms",   mlp="swiglu", bias=False, n_kv_heads < n_heads (grouped-query attention)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class BlockConfig:
    dim: int = 128
    n_heads: int = 4
    n_kv_heads: int | None = None  # None = full multi-head attention
    mlp_hidden: int | None = None  # None = 4 * dim
    mlp: str = "gelu"  # "gelu" (two matrices) or "swiglu" (three)
    norm: str = "layer"  # "layer" or "rms"
    bias: bool = True
    prenorm: bool = True  # False = the 2017 post-norm block
    layerscale: float | None = None  # ViT-side: per-channel gain on each branch, started near zero

    @property
    def hidden(self) -> int:
        return self.mlp_hidden or 4 * self.dim


def make_norm(cfg: BlockConfig) -> nn.Module:
    return nn.LayerNorm(cfg.dim) if cfg.norm == "layer" else nn.RMSNorm(cfg.dim, eps=1e-6)


class Attention(nn.Module):
    """Project to Q, K, V; split heads; score; scale by 1/sqrt(d_h); mask; softmax; weight V; merge; project."""

    def __init__(self, cfg: BlockConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv = cfg.n_kv_heads or cfg.n_heads
        self.head_dim = cfg.dim // cfg.n_heads
        self.q = nn.Linear(cfg.dim, self.n_heads * self.head_dim, bias=cfg.bias)
        self.k = nn.Linear(cfg.dim, self.n_kv * self.head_dim, bias=cfg.bias)
        self.v = nn.Linear(cfg.dim, self.n_kv * self.head_dim, bias=cfg.bias)
        self.out = nn.Linear(self.n_heads * self.head_dim, cfg.dim, bias=cfg.bias)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None, cache: dict | None = None) -> torch.Tensor:
        """x (B, N, d). mask: (N, Nk) or (B, N, Nk) bool, True = may attend; None = everyone sees everyone.

        cache: a dict that keeps K and V of earlier calls. New keys are appended, old ones are never recomputed.
        """
        b, n, _ = x.shape
        q = self.q(x).view(b, n, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, N, dh)
        k = self.k(x).view(b, n, self.n_kv, self.head_dim).transpose(1, 2)
        v = self.v(x).view(b, n, self.n_kv, self.head_dim).transpose(1, 2)
        if cache is not None:
            if "k" in cache:
                k, v = torch.cat([cache["k"], k], dim=2), torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v
        if self.n_kv != self.n_heads:  # grouped-query: each K/V head serves n_heads / n_kv query heads
            k = k.repeat_interleave(self.n_heads // self.n_kv, dim=1)
            v = v.repeat_interleave(self.n_heads // self.n_kv, dim=1)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)  # (B, H, N, Nk)
        if mask is not None:
            m = mask if mask.dim() == 2 else mask[:, None]
            scores = scores.masked_fill(~m, float("-inf"))
        weights = scores.softmax(dim=-1)
        y = (weights @ v).transpose(1, 2).reshape(b, n, -1)
        return self.out(y)


class MLP(nn.Module):
    """Expand, nonlinearity, contract. SwiGLU adds a gate matrix: silu(x W_g) * (x W_u), then W_d."""

    def __init__(self, cfg: BlockConfig) -> None:
        super().__init__()
        self.kind = cfg.mlp
        self.up = nn.Linear(cfg.dim, cfg.hidden, bias=cfg.bias)
        self.gate = nn.Linear(cfg.dim, cfg.hidden, bias=cfg.bias) if cfg.mlp == "swiglu" else None
        self.down = nn.Linear(cfg.hidden, cfg.dim, bias=cfg.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.gate(x)) * self.up(x) if self.gate is not None else F.gelu(self.up(x))
        return self.down(h)


class Block(nn.Module):
    def __init__(self, cfg: BlockConfig) -> None:
        super().__init__()
        self.prenorm = cfg.prenorm
        self.norm1, self.attn = make_norm(cfg), Attention(cfg)
        self.norm2, self.mlp = make_norm(cfg), MLP(cfg)
        if cfg.layerscale is not None:
            self.gamma1 = nn.Parameter(torch.full((cfg.dim,), cfg.layerscale))
            self.gamma2 = nn.Parameter(torch.full((cfg.dim,), cfg.layerscale))
        else:
            self.gamma1 = self.gamma2 = None

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None, cache: dict | None = None) -> torch.Tensor:
        g1 = self.gamma1 if self.gamma1 is not None else 1.0
        g2 = self.gamma2 if self.gamma2 is not None else 1.0
        if self.prenorm:  # an unbroken identity path from input to output
            x = x + g1 * self.attn(self.norm1(x), mask, cache)
            return x + g2 * self.mlp(self.norm2(x))
        x = self.norm1(x + g1 * self.attn(x, mask, cache))  # post-norm: the norm sits on the main path
        return self.norm2(x + g2 * self.mlp(x))
