"""Five connectors from "Fewer tokens by design", and a toy task that shows what each one keeps.

Pixel shuffle is imported from C7's `tinyvlm` (the Idefics3 / SmolVLM version), not rebuilt here.
The toy: an 8 x 8 grid of patch features. One task depends on the arrangement inside a 2 x 2 block (like a small
glyph); the other depends only on where a bright blob is (like a scene question). A connector shrinks the grid,
and a one-layer transformer reads the result. This is a mechanism demo, not a model of any real VLM.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "c07-vlm"))
from tinyvlm.projector import pixel_shuffle  # noqa: E402  (C7: fold f x f patches into channels)

SIDE, DIM = 8, 8


def unshuffle(tokens: torch.Tensor, factor: int) -> torch.Tensor:
    """Inverse of `pixel_shuffle`: proof that the fold drops no number."""
    n, m, d = tokens.shape
    side, dim = int(round(m ** 0.5)) * factor, d // (factor * factor)
    x = tokens.reshape(n, side // factor, side // factor, d).permute(0, 2, 1, 3)
    x = x.reshape(n, side // factor, side, dim * factor).permute(0, 2, 1, 3)
    return x.reshape(n, side * side, dim)


def avg_pool(tokens: torch.Tensor, factor: int) -> torch.Tensor:
    """Gemma 3 style: the mean of each factor x factor block."""
    n, m, d = tokens.shape
    side = int(round(m ** 0.5))
    grid = tokens.transpose(1, 2).reshape(n, d, side, side)
    return nn.functional.avg_pool2d(grid, factor).flatten(2).transpose(1, 2)


class Connector(nn.Module):
    """kind: mlp (1 token per patch), shuffle, pool, resampler (learned queries) or conv (strided 3 x 3)."""

    def __init__(self, kind: str, dim: int = DIM, out: int = 32, factor: int = 2, queries: int = 4):
        super().__init__()
        self.kind, self.factor = kind, factor
        d_in = dim * factor * factor if kind == "shuffle" else dim
        if kind == "resampler":
            self.queries = nn.Parameter(torch.randn(queries, out))
            self.kv = nn.Linear(dim, out)
            self.pos = nn.Parameter(torch.randn(SIDE * SIDE, out) * 0.02)
            self.attn = nn.MultiheadAttention(out, 4, batch_first=True)
        elif kind == "conv":
            self.conv = nn.Conv2d(dim, out, 3, stride=factor, padding=1)
        if kind != "resampler":
            self.proj = nn.Sequential(nn.Linear(out if kind == "conv" else d_in, out), nn.GELU(), nn.Linear(out, out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # n x 64 x dim -> n x tokens x out
        if self.kind == "mlp":
            return self.proj(x)
        if self.kind == "shuffle":
            return self.proj(pixel_shuffle(x, self.factor))
        if self.kind == "pool":
            return self.proj(avg_pool(x, self.factor))
        if self.kind == "resampler":
            kv = self.kv(x) + self.pos
            q = self.queries.expand(x.shape[0], -1, -1)
            return self.attn(q, kv, kv, need_weights=False)[0]
        grid = x.transpose(1, 2).reshape(x.shape[0], -1, SIDE, SIDE)
        return self.proj(self.conv(grid).flatten(2).transpose(1, 2))


class Reader(nn.Module):
    """A one-layer transformer over the connector's tokens, standing in for the language model."""

    def __init__(self, connector: Connector, out: int = 32, classes: int = 4):
        super().__init__()
        self.connector = connector
        self.pos = nn.Parameter(torch.randn(SIDE * SIDE, out) * 0.02)
        self.cls = nn.Parameter(torch.zeros(1, 1, out))
        self.block = nn.TransformerEncoderLayer(out, 4, 64, dropout=0.0, batch_first=True)
        self.head = nn.Linear(out, classes)

    def forward(self, x):
        t = self.connector(x)
        t = t + self.pos[: t.shape[1]]
        t = torch.cat([self.cls.expand(len(t), -1, -1), t], 1)
        return self.head(self.block(t)[:, 0])


def make_task(task: str, n: int, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Patch grids n x 64 x DIM and labels in {0..3}.

    glyph: one fixed 2 x 2 block holds a "stroke" in one of its four cells; the label is which cell.
           Block means are identical for all four labels, so a 2 x 2 average cannot see it.
    blob:  a bright 2 x 2 blob sits in one of the four quadrants; the label is the quadrant.
    """
    x = torch.randn(n, SIDE, SIDE, DIM, generator=generator) * 0.3
    y = torch.randint(0, 4, (n,), generator=generator)
    for i in range(n):
        if task == "glyph":
            br, bc = 2, 4                                     # the "sign": a fixed 2 x 2 block
            r, c = divmod(int(y[i]), 2)
            x[i, br + r, bc + c, 0] += 4.0                    # the stroke; block mean is the same for every label
        else:
            qr, qc = divmod(int(y[i]), 2)
            r = qr * 4 + int(torch.randint(0, 3, (1,), generator=generator))
            c = qc * 4 + int(torch.randint(0, 3, (1,), generator=generator))
            x[i, r:r + 2, c:c + 2, 1] += 2.0
    return x.reshape(n, SIDE * SIDE, DIM), y


def train_and_eval(kind: str, task: str, steps: int = 400, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    xtr, ytr = make_task(task, 2048, g)
    xte, yte = make_task(task, 512, g)
    model = Reader(Connector(kind))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for step in range(steps):
        idx = torch.randint(0, len(xtr), (128,), generator=g)
        loss = nn.functional.cross_entropy(model(xtr[idx]), ytr[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        acc = (model(xte).argmax(-1) == yte).float().mean().item()
        tokens = model.connector(xte[:1]).shape[1]
    return {"connector": kind, "task": task, "tokens": tokens, "test_accuracy": round(acc, 3)}
