"""Where the bytes go: count main-memory (HBM) traffic for the textbook kernel and for a tiled one.

Model: tensors live in slow memory. A kernel may hold a few tiles in fast memory (SRAM). Every
element moved between the two is counted. Both kernels below really compute attention, through a
counter that sees every load and store, so the counts belong to code that returns the right answer.

  textbook  S = Q K^T is written out, read back for the softmax, P is written, read back for P V.
            Traffic is about 4 n^2 + 4 n d per head.
  tiled     A block of B queries stays on chip while every block of K and V streams past. Q and O move
            once, K and V move once per query block: about 2 n d + 2 n^2 d / B per head.
            The score matrix never exists in slow memory.
"""

from __future__ import annotations

import math

import torch


class SlowMemory:
    """Counts elements read from and written to slow memory."""

    def __init__(self) -> None:
        self.reads = self.writes = 0

    def load(self, t: torch.Tensor) -> torch.Tensor:
        self.reads += t.numel()
        return t.clone()

    def store(self, t: torch.Tensor) -> torch.Tensor:
        self.writes += t.numel()
        return t.clone()

    @property
    def total(self) -> int:
        return self.reads + self.writes


def textbook_attention(q, k, v, mem: SlowMemory) -> torch.Tensor:
    """Three kernels with the (n, n) intermediates written out between them. q, k, v: (n, d)."""
    scale = 1.0 / math.sqrt(q.size(-1))
    s = mem.store(mem.load(q) @ mem.load(k).T * scale)  # kernel 1: scores
    p = mem.store(torch.softmax(mem.load(s), dim=-1))  # kernel 2: softmax
    return mem.store(mem.load(p) @ mem.load(v))  # kernel 3: weighted sum


def tiled_attention(q, k, v, mem: SlowMemory, block: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    """One kernel with an online softmax. Returns the output and the log-sum-exp kept for backward."""
    n, d = q.shape
    scale = 1.0 / math.sqrt(d)
    out, lse = torch.empty_like(q), torch.empty(n, dtype=q.dtype)
    for qs in range(0, n, block):
        qt = mem.load(q[qs : qs + block]) * scale
        m = torch.full((qt.size(0), 1), float("-inf"), dtype=q.dtype)  # running max, on chip
        l = torch.zeros(qt.size(0), 1, dtype=q.dtype)  # running denominator, on chip
        acc = torch.zeros(qt.size(0), v.size(1), dtype=q.dtype)  # running output, on chip
        for ks in range(0, n, block):
            kt, vt = mem.load(k[ks : ks + block]), mem.load(v[ks : ks + block])
            s = qt @ kt.T
            m_new = torch.maximum(m, s.amax(-1, keepdim=True))
            p = torch.exp(s - m_new)
            alpha = torch.exp(m - m_new)  # rescale what was summed against the old max
            l, acc, m = alpha * l + p.sum(-1, keepdim=True), alpha * acc + p @ vt, m_new
        out[qs : qs + block] = mem.store(acc / l)
        lse[qs : qs + block] = mem.store((m + l.log())[:, 0])  # n numbers: all backward needs to rebuild P
    return out, lse


def textbook_formula(n: int, d: int) -> int:
    return 4 * n * n + 4 * n * d


def tiled_formula(n: int, d: int, block: int) -> int:
    blocks = math.ceil(n / block)
    return n * d + blocks * 2 * n * d + n * d + n  # Q once, K and V per query block, O once, lse once


def traffic_table(ns: list[int], d: int, blocks: list[int]) -> list[dict]:
    rows = []
    for n in ns:
        g = torch.Generator().manual_seed(n)
        q, k, v = (torch.randn(n, d, generator=g, dtype=torch.float64) for _ in range(3))
        mem = SlowMemory()
        ref = textbook_attention(q, k, v, mem)
        row = {"n": n, "textbook": mem.total, "textbook formula": textbook_formula(n, d)}
        for b in blocks:
            mt = SlowMemory()
            out, _ = tiled_attention(q, k, v, mt, block=b)
            row[f"tiled B={b}"] = mt.total
            row[f"tiled B={b} formula"] = tiled_formula(n, d, b)
            row[f"max |diff| B={b}"] = (out - ref).abs().max().item()
        rows.append(row)
    return rows


def saved_for_backward(n: int, d: int) -> dict:
    """Numbers per head kept between forward and backward."""
    return {"textbook (P, n^2)": n * n, "tiled (O and log-sum-exp, n d + n)": n * d + n}


def reassociation_drift(n: int = 512, d: int = 64, blocks=(16, 64, 256)) -> dict:
    """Same function, different summation order: fp32 results differ only at rounding level."""
    g = torch.Generator().manual_seed(0)
    q, k, v = (torch.randn(n, d, generator=g) for _ in range(3))
    ref = textbook_attention(q, k, v, SlowMemory())
    ref64 = textbook_attention(q.double(), k.double(), v.double(), SlowMemory())
    out = {"textbook fp32 vs fp64": (ref.double() - ref64).abs().max().item()}
    for b in blocks:
        out[f"tiled B={b} vs textbook, fp32"] = (tiled_attention(q, k, v, SlowMemory(), b)[0] - ref).abs().max().item()
    return out
