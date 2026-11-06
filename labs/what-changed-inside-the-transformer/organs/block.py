"""One decoder whose every organ can be swapped: 2017 parts or 2026 parts.

    norm      "layer" (LayerNorm) or "rms" (RMSNorm, from C4's tinyllm)
    placement "post"      x = Norm(x + F(x))                  2017
              "pre"       x = x + F(Norm(x))                  GPT-2 onwards
              "sandwich"  x = x + Norm(F(Norm(x)))            Gemma 2 and 3
              "postbranch" x = x + Norm(F(x))                 Olmo 2 and 3
    pos       "sinusoid" | "learned" | "rope" | "nope"
    ffn       "gelu" (two matrices, 4d) or "swiglu" (three matrices, C4's SwiGLU)
    qk_norm   RMSNorm on q and k per head before the dot product

Only the parts the post compares are configurable. Attention is plain causal multi-head attention.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from . import _paths  # noqa: F401
from tinyllm.mlp import SwiGLU
from tinyllm.norm import RMSNorm
from tinyllm.rope import RotaryEmbedding, rotate


@dataclass
class BlockConfig:
    vocab: int = 256
    d: int = 128
    layers: int = 6
    heads: int = 4
    max_seq: int = 256
    norm: str = "rms"
    placement: str = "pre"
    pos: str = "rope"
    ffn: str = "swiglu"
    qk_norm: bool = False
    rope_base: float = 10_000.0


def make_norm(kind: str, d: int) -> nn.Module:
    return nn.LayerNorm(d) if kind == "layer" else RMSNorm(d)


def sinusoid_table(seq: int, d: int) -> torch.Tensor:
    """p[m, 2i] = sin(m / 10000^(2i/d)), p[m, 2i+1] = cos(same). Vaswani et al., 2017."""
    pos = torch.arange(seq, dtype=torch.float32)[:, None]
    inv = 10_000.0 ** (-torch.arange(0, d, 2, dtype=torch.float32) / d)
    table = torch.zeros(seq, d)
    table[:, 0::2], table[:, 1::2] = torch.sin(pos * inv), torch.cos(pos * inv)
    return table


class GeluFFN(nn.Module):
    """The 2017 / GPT-2 FFN: W2 gelu(W1 x), width 4d, 8 d^2 parameters."""

    def __init__(self, d: int, width: int) -> None:
        super().__init__()
        self.up, self.down = nn.Linear(d, width, bias=False), nn.Linear(width, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.gelu(self.up(x)))


class Attention(nn.Module):
    def __init__(self, c: BlockConfig) -> None:
        super().__init__()
        self.h, self.dh = c.heads, c.d // c.heads
        self.qkv = nn.Linear(c.d, 3 * c.d, bias=False)
        self.out = nn.Linear(c.d, c.d, bias=False)
        self.rope = RotaryEmbedding(self.dh, c.max_seq, c.rope_base) if c.pos == "rope" else None
        self.q_norm = RMSNorm(self.dh) if c.qk_norm else None
        self.k_norm = RMSNorm(self.dh) if c.qk_norm else None
        self.last_logit_max = 0.0  # largest |q.k / sqrt(d_h)| of the last call, for the QK-norm demo

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        q, k, v = self.qkv(x).view(b, t, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)  # each (B, H, T, dh)
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        if self.rope is not None:
            cos, sin = self.rope(torch.arange(t, device=x.device)[None].expand(b, t))
            q, k = rotate(q, cos, sin), rotate(k, cos, sin)
        with torch.no_grad():
            self.last_logit_max = (q @ k.transpose(-1, -2)).abs().max().item() / math.sqrt(self.dh)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(y.transpose(1, 2).reshape(b, t, -1))


class Block(nn.Module):
    def __init__(self, c: BlockConfig) -> None:
        super().__init__()
        self.placement = c.placement
        self.attn = Attention(c)
        self.ffn = SwiGLU(c.d, 64 * math.ceil(8 * c.d / 3 / 64)) if c.ffn == "swiglu" else GeluFFN(c.d, 4 * c.d)
        self.n1, self.n2 = make_norm(c.norm, c.d), make_norm(c.norm, c.d)
        if c.placement == "sandwich":  # a second norm on each branch output
            self.n1b, self.n2b = make_norm(c.norm, c.d), make_norm(c.norm, c.d)

    def _sub(self, x: torch.Tensor, f: nn.Module, n: nn.Module, nb: nn.Module | None) -> torch.Tensor:
        p = self.placement
        if p == "post":
            return n(x + f(x))
        if p == "pre":
            return x + f(n(x))
        if p == "sandwich":
            return x + nb(f(n(x)))
        if p == "postbranch":
            return x + n(f(x))
        raise ValueError(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sandwich = self.placement == "sandwich"
        x = self._sub(x, self.attn, self.n1, self.n1b if sandwich else None)
        return self._sub(x, self.ffn, self.n2, self.n2b if sandwich else None)


class Decoder(nn.Module):
    """Embed, add a position table if the config asks for one, loop over blocks, final norm, LM head."""

    def __init__(self, c: BlockConfig) -> None:
        super().__init__()
        self.c = c
        self.emb = nn.Embedding(c.vocab, c.d)
        if c.pos == "learned":
            self.pos = nn.Parameter(torch.randn(c.max_seq, c.d) * 0.02)
        elif c.pos == "sinusoid":
            self.register_buffer("pos", sinusoid_table(c.max_seq, c.d), persistent=False)
        else:
            self.pos = None
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.layers))
        # Post-LN already ends on a norm. Every other placement needs a final one.
        self.final = nn.Identity() if c.placement == "post" else make_norm(c.norm, c.d)
        self.head = nn.Linear(c.d, c.vocab, bias=False)
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, ids: torch.Tensor, trace: list | None = None) -> torch.Tensor:
        x = self.emb(ids)
        if self.pos is not None:
            x = x + self.pos[: ids.shape[1]]
        if trace is not None:
            trace.append(x)
        for blk in self.blocks:
            x = blk(x)
            if trace is not None:
                trace.append(x)
        return self.head(self.final(x))
