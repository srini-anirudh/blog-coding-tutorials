"""A tiny causal transformer whose only knob is where the norm sits (post sections 07 and 08).

Placements, for one sublayer F (attention or MLP) applied to the residual stream x at layer l (1-based):
  post      x <- Norm(x + F(x))                    Vaswani et al. 2017
  pre       x <- x + F(Norm(x))                    GPT-2 through Llama 3
  peri      x <- x + Norm(F(Norm(x)))              Sandwich-LN / Peri-LN / Gemma 2-3
  lns       x <- x + F(Norm(x) / sqrt(l))          LayerNorm Scaling, Sun et al. 2025
  deepnorm  x <- Norm(alpha * x + F(x)), F init at gain beta, alpha = (2N)^1/4, beta = (8N)^-1/4
  dyt       pre, with every norm replaced by Dynamic Tanh
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .ops import make_norm

PLACEMENTS = ("post", "pre", "peri", "lns", "deepnorm", "dyt")


class Attention(nn.Module):
    def __init__(self, d: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q, k, v = self.qkv(x).view(b, t, 3, self.n_heads, d // self.n_heads).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(y.transpose(1, 2).reshape(b, t, d))


class MLP(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.up = nn.Linear(d, 4 * d, bias=False)
        self.down = nn.Linear(4 * d, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.gelu(self.up(x)))


class Block(nn.Module):
    def __init__(self, d: int, n_heads: int, placement: str, layer: int, n_layers: int):
        super().__init__()
        assert placement in PLACEMENTS, placement
        self.placement, self.layer = placement, layer
        kind = "dyt" if placement == "dyt" else "rms"
        self.attn, self.mlp = Attention(d, n_heads), MLP(d)
        self.norm_in = nn.ModuleList([make_norm(kind, d), make_norm(kind, d)])
        self.norm_out = nn.ModuleList([make_norm(kind, d), make_norm(kind, d)]) if placement == "peri" else None
        self.alpha = (2 * n_layers) ** 0.25 if placement == "deepnorm" else 1.0
        if placement == "deepnorm":  # DeepNet: shrink the value, output and MLP weights by beta at init
            beta = (8 * n_layers) ** -0.25
            with torch.no_grad():
                self.attn.qkv.weight[2 * d :].mul_(beta)
                for w in (self.attn.out.weight, self.mlp.up.weight, self.mlp.down.weight):
                    w.mul_(beta)

    def sublayer(self, x: torch.Tensor, f: nn.Module, i: int) -> torch.Tensor:
        p, n = self.placement, self.norm_in[i]
        if p == "post":
            return n(x + f(x))
        if p == "deepnorm":
            return n(self.alpha * x + f(x))
        if p in ("pre", "dyt"):
            return x + f(n(x))
        if p == "lns":
            return x + f(n(x) / math.sqrt(self.layer))
        return x + self.norm_out[i](f(n(x)))  # peri

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sublayer(self.sublayer(x, self.attn, 0), self.mlp, 1)


class TinyLM(nn.Module):
    """Character-level causal LM. forward returns logits and the residual stream after every block."""

    def __init__(self, vocab: int, d: int = 64, n_layers: int = 24, n_heads: int = 4, placement: str = "pre", seq: int = 64):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(seq, d)
        self.blocks = nn.ModuleList(Block(d, n_heads, placement, l + 1, n_layers) for l in range(n_layers))
        # Post-LN and DeepNorm already end on a norm. The others need a final norm before the head.
        final = placement not in ("post", "deepnorm")
        self.final_norm = make_norm("dyt" if placement == "dyt" else "rms", d) if final else nn.Identity()
        self.head = nn.Linear(d, vocab, bias=False)
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
        self.skip: set[int] = set()  # block indices to bypass, for layer-removal experiments

    def forward(self, idx: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x = self.tok(idx) + self.pos(torch.arange(idx.shape[1], device=idx.device))
        stream = [x]
        for i, block in enumerate(self.blocks):
            if i not in self.skip:
                x = block(x)
            stream.append(x)
        return self.head(self.final_norm(x)), stream


def lm_loss(model: TinyLM, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    logits, _ = model(x)
    return F.cross_entropy(logits.flatten(0, 1), y.flatten())
