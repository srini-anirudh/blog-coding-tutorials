"""Context parallelism: ring attention and Ulysses, on a sequence split across C ranks.

Ring attention keeps queries local and passes key/value blocks around the ring. Each block's scores are folded
into a running max m, a running denominator l and a running output o (online softmax), so the result is exact.
The exchange for the next step is posted before the current block is computed, which is the overlap.

Ulysses turns a sequence-sharded, all-heads layout into a head-sharded, full-sequence layout with one All-to-All
per tensor, runs ordinary attention, and transposes back.

The ring here is forward only. A training implementation also needs the ring backward pass.
"""

from __future__ import annotations

import math

import torch
import torch.distributed as dist

from . import comm
from .model import attention


def positions(s: int, C: int, rank: int, layout: str) -> torch.Tensor:
    """Global token positions held by `rank`. contiguous: one block of s / C. zigzag: chunk r and chunk
    2C - 1 - r of 2C chunks, so every rank holds an early piece and a late piece of the causal triangle."""
    if layout == "contiguous":
        return torch.arange(rank * s // C, (rank + 1) * s // C)
    c = s // (2 * C)
    return torch.cat([torch.arange(rank * c, (rank + 1) * c), torch.arange((2 * C - 1 - rank) * c, (2 * C - rank) * c)])


def shard_seq(x: torch.Tensor, C: int, rank: int, layout: str, dim: int = 2) -> torch.Tensor:
    return x.index_select(dim, positions(x.shape[dim], C, rank, layout).to(x.device)).contiguous()


def _pieces(pos: torch.Tensor, layout: str):
    """Sub-blocks of a rank's tokens that are contiguous in the sequence: 1 for contiguous, 2 for zigzag."""
    n = 1 if layout == "contiguous" else 2
    step = pos.numel() // n
    return [slice(i * step, (i + 1) * step) for i in range(n)]


def ring_attention(q, k, v, q_pos, layout: str = "contiguous", causal: bool = True, overlap: bool = True):
    """q, k, v: [b, heads, s_local, d] on this rank; q_pos: this rank's global positions.
    Returns (output [b, heads, s_local, d], number of score blocks this rank computed)."""
    rank, C = dist.get_rank(), dist.get_world_size()
    scale = 1 / math.sqrt(q.shape[-1])
    m = torch.full(q.shape[:-1] + (1,), float("-inf"), device=q.device, dtype=torch.float32)
    l = torch.zeros_like(m)
    o = torch.zeros(q.shape, device=q.device, dtype=torch.float32)
    kv, kv_pos = torch.stack([k, v]).contiguous(), q_pos.to(q.device)
    blocks = 0
    q_pieces = _pieces(q_pos, layout)
    for step in range(C):
        reqs = []
        if step < C - 1:  # post the rotation now; it travels while this block computes
            nxt, nxt_pos = torch.empty_like(kv), torch.empty_like(kv_pos)
            ops = [dist.P2POp(dist.isend, kv, (rank + 1) % C), dist.P2POp(dist.irecv, nxt, (rank - 1) % C),
                   dist.P2POp(dist.isend, kv_pos, (rank + 1) % C), dist.P2POp(dist.irecv, nxt_pos, (rank - 1) % C)]
            comm._log("ring_kv_send", kv)
            reqs = dist.batch_isend_irecv(ops)
            if not overlap:
                for r in reqs:
                    r.wait()
        for qs in q_pieces:
            for ks in _pieces(kv_pos, layout):
                qp, kp = q_pos[qs].to(q.device), kv_pos[ks]
                if causal and kp.min() > qp.max():
                    continue  # this block lies entirely above the diagonal: nothing to compute
                blocks += 1
                s = (q[:, :, qs].float() @ kv[0][:, :, ks].float().transpose(-1, -2)) * scale
                if causal:
                    s = s.masked_fill(kp[None, :] > qp[:, None], float("-inf"))
                m_old = m[:, :, qs]
                m_new = torch.maximum(m_old, s.amax(-1, keepdim=True))
                p = torch.exp(s - m_new)
                corr = torch.exp(m_old - m_new)
                l[:, :, qs] = l[:, :, qs] * corr + p.sum(-1, keepdim=True)
                o[:, :, qs] = o[:, :, qs] * corr + p @ kv[1][:, :, ks].float()
                m[:, :, qs] = m_new
        for r in reqs:
            r.wait()
        if step < C - 1:
            kv, kv_pos = nxt, nxt_pos
    return (o / l).to(q.dtype), blocks


def ulysses_attention(q, k, v, causal: bool = True, group=None):
    """q, k, v: [b, heads, s / C, d], contiguous sequence shards. Four All-to-Alls: q, k, v in, output back."""
    assert q.shape[1] % dist.get_world_size(group) == 0, "Ulysses needs the head count to divide by C"
    q, k, v = (comm.a2a(t, 1, 2, group) for t in (q, k, v))  # scatter heads, gather sequence
    o = attention(q, k, v, causal)  # unmodified attention over the full sequence, H / C heads
    return comm.a2a(o, 2, 1, group)  # scatter sequence, gather heads
