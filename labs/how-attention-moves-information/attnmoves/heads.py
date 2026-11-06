"""Why more than one head: one softmax is one notion of relevance.

Task: at position i, output two things, the token at i - 1 and the token at i - 3 (cyclic).
A single head has one distribution per position. If it splits its weight between i - 1 and i - 3,
the value it returns is a sum, W_V (x_{i-1} + x_{i-3}), and a linear sum forgets which token sat
where. Two heads can each take one lookup. All models below have the same parameter count,
because d_h = d_model / h: heads partition the width, they do not add to it.
"""

from __future__ import annotations

import itertools

import torch
from torch import nn

from .common import attention

OFFSETS = (1, 3)


class MultiHeadRouter(nn.Module):
    def __init__(self, vocab: int, n: int, d: int = 32, heads: int = 1) -> None:
        super().__init__()
        assert d % heads == 0
        self.h, self.dh = heads, d // heads
        self.tok, self.pos = nn.Embedding(vocab, d), nn.Embedding(n, d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.readout = nn.ModuleList(nn.Linear(d, vocab) for _ in OFFSETS)
        self.head_gate = torch.ones(heads)  # set entries to 0 to ablate heads

    def forward(self, tokens: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        b, n = tokens.shape
        x = self.tok(tokens) + self.pos.weight[:n]
        q, k, v = self.qkv(x).view(b, n, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)  # each (B, H, N, dh)
        z, w = attention(q, k, v)  # (B, H, N, dh), (B, H, N, N)
        z = z * self.head_gate[None, :, None, None]
        z = self.o(z.transpose(1, 2).reshape(b, n, -1))  # concatenate heads, then mix with W_O
        return [r(z) for r in self.readout], w


def batch(size: int, n: int, vocab: int, g: torch.Generator) -> tuple[torch.Tensor, list[torch.Tensor]]:
    tokens = torch.randint(vocab, (size, n), generator=g)
    return tokens, [tokens.roll(off, dims=1) for off in OFFSETS]


def train(heads: int, n: int = 12, vocab: int = 12, steps: int = 2000, seed: int = 0) -> tuple[MultiHeadRouter, dict]:
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    model = MultiHeadRouter(vocab, n, heads=heads)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(steps):
        tokens, targets = batch(256, n, vocab, g)
        logits, _ = model(tokens)
        loss = sum(nn.functional.cross_entropy(lg.reshape(-1, vocab), t.reshape(-1)) for lg, t in zip(logits, targets))
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model, evaluate(model, n, vocab, seed)


@torch.no_grad()
def evaluate(model: MultiHeadRouter, n: int = 12, vocab: int = 12, seed: int = 0) -> dict:
    tokens, targets = batch(4096, n, vocab, torch.Generator().manual_seed(seed + 999))
    logits, w = model(tokens)
    idx = torch.arange(n)
    out = {f"acc(i-{off})": (lg.argmax(-1) == t).float().mean().item() for off, lg, t in zip(OFFSETS, logits, targets)}
    for off in OFFSETS:  # where each head looks: mean weight on the two target offsets
        out[f"weight on i-{off} per head"] = [round(w[:, h, idx, (idx - off) % n].mean().item(), 2) for h in range(model.h)]
    out["params"] = sum(p.numel() for p in model.parameters())
    return out


def ablate(model: MultiHeadRouter, keep: int, n: int = 12, vocab: int = 12) -> list[dict]:
    """Keep only `keep` heads, for every choice of heads. Returns accuracies for each subset."""
    rows = []
    for subset in itertools.combinations(range(model.h), keep):
        model.head_gate = torch.zeros(model.h)
        model.head_gate[list(subset)] = 1.0
        res = evaluate(model, n, vocab)
        rows.append({"heads kept": list(subset), **{k: v for k, v in res.items() if k.startswith("acc")}})
    model.head_gate = torch.ones(model.h)
    return rows
