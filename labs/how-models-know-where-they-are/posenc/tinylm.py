"""A tiny transformer whose only moving part is the positional scheme.

pe = "none"        NoPE: nothing added, nothing rotated. Order can enter only through the causal mask.
pe = "learned"     a learned table P[p] added to the token embedding (GPT-2, BERT). Hard length limit.
pe = "sinusoidal"  the fixed frequency code added to the token embedding (Vaswani et al.).
pe = "rope"        queries and keys rotated by position (half-split, from C3).
pe = "alibi"       -slope_h * |i - j| added to the logits (from C3).

At evaluation, `rope_freqs` and `logit_scale` let a RoPE model run with PI, NTK or YaRN without retraining.
"""

from __future__ import annotations

import math
from typing import Callable

import torch
from torch import nn
from torch.nn import functional as F

from .c03 import alibi_bias, alibi_slopes, rotate
from .rope import cos_sin, inv_freq
from .sinusoidal import sinusoidal

PE_KINDS = ("none", "learned", "sinusoidal", "rope", "alibi")


class Attention(nn.Module):
    def __init__(self, d: int, heads: int) -> None:
        super().__init__()
        self.h, self.dh = heads, d // heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x, causal, rope=None, bias=None, logit_scale=1.0, return_attn=False):
        B, N, _ = x.shape
        q, k, v = self.qkv(x).view(B, N, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)  # each (B, H, N, dh)
        if rope is not None:
            cos, sin = rope
            q, k = rotate(q, cos, sin), rotate(k, cos, sin)
        logits = (q @ k.transpose(-1, -2)) * (logit_scale / math.sqrt(self.dh))
        if bias is not None:
            logits = logits + bias
        if causal:
            logits = logits.masked_fill(torch.ones(N, N, dtype=torch.bool, device=x.device).triu(1), float("-inf"))
        attn = logits.softmax(-1)
        y = self.out((attn @ v).transpose(1, 2).reshape(B, N, -1))
        return (y, attn) if return_attn else y


class Block(nn.Module):
    def __init__(self, d: int, heads: int) -> None:
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = Attention(d, heads)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, **kw):
        x = x + self.attn(self.n1(x), **kw)
        return x + self.mlp(self.n2(x))


class TinyLM(nn.Module):
    def __init__(
        self, vocab: int, n_out: int | None = None, d: int = 128, heads: int = 4, layers: int = 2,
        pe: str = "rope", max_len: int = 64, causal: bool = True, base: float = 10_000.0,
    ) -> None:
        super().__init__()
        if pe not in PE_KINDS:
            raise ValueError(f"pe must be one of {PE_KINDS}")
        self.pe, self.causal, self.max_len, self.heads = pe, causal, max_len, heads
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_len, d) if pe == "learned" else None
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, n_out or vocab)
        self.d, self.head_dim = d, d // heads
        self.register_buffer("rope_freqs", inv_freq(d // heads, base), persistent=False)
        self.register_buffer("slopes", alibi_slopes(heads), persistent=False)

    def forward(self, tokens: torch.Tensor, rope_freqs: torch.Tensor | None = None, logit_scale: float = 1.0):
        B, N = tokens.shape
        x = self.tok(tokens)
        pos = torch.arange(N, device=tokens.device)
        if self.pe == "learned":
            if N > self.max_len:
                raise ValueError(f"learned table has {self.max_len} rows; position {N - 1} does not exist")
            x = x + self.pos(pos)
        elif self.pe == "sinusoidal":
            x = x + sinusoidal(pos, self.d).float()
        rope = bias = None
        if self.pe == "rope":
            cos, sin = cos_sin(pos, self.rope_freqs if rope_freqs is None else rope_freqs)
            rope = (cos[None], sin[None])
        elif self.pe == "alibi":
            bias = alibi_bias(self.slopes, N, N)
        for blk in self.blocks:
            x = blk(x, causal=self.causal, rope=rope, bias=bias, logit_scale=logit_scale)
        return self.head(self.norm(x))


def train(
    model: nn.Module, batch_fn: Callable[[torch.Generator], tuple], steps: int, lr: float = 1e-3,
    seed: int = 0, log_every: int = 0, log: Callable[[str], None] = print,
) -> list[float]:
    """AdamW with cosine decay. batch_fn(gen) -> (tokens, targets, mask). Loss is cross entropy on mask."""
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=max(0.05, 3 / steps))
    losses = []
    model.train()
    for step in range(steps):
        tokens, targets, mask = batch_fn(gen)
        logits = model(tokens)
        loss = F.cross_entropy(logits[mask], targets[mask])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        losses.append(loss.item())
        if log_every and (step + 1) % log_every == 0:
            log(f"  step {step + 1:5d}  loss {sum(losses[-log_every:]) / log_every:.3f}")
    model.eval()
    return losses


@torch.no_grad()
def accuracy(model: nn.Module, batch_fn, batches: int = 10, seed: int = 1234, **fwd) -> float:
    gen = torch.Generator().manual_seed(seed)
    right = total = 0
    for _ in range(batches):
        tokens, targets, mask = batch_fn(gen)
        pred = model(tokens, **fwd).argmax(-1)
        right += int((pred[mask] == targets[mask]).sum())
        total += int(mask.sum())
    return right / total
