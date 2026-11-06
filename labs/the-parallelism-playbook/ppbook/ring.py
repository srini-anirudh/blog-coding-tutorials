"""Ring ReduceScatter, ring AllGather, and AllReduce as the two in sequence, built from send/recv.

With N ranks the buffer is cut into N chunks. In ReduceScatter, at step k rank r sends chunk (r - k) mod N to
its right neighbour and adds the chunk arriving from its left. After N - 1 steps rank r owns the full sum of
chunk (r + 1) mod N. AllGather then circulates the finished chunks for another N - 1 steps.
Each rank sends 2 (N - 1) / N of the buffer in total, whatever N is.
"""

from __future__ import annotations

import torch
import torch.distributed as dist


def _exchange(send: torch.Tensor, recv: torch.Tensor, rank: int, n: int) -> None:
    right, left = (rank + 1) % n, (rank - 1) % n
    reqs = dist.batch_isend_irecv([dist.P2POp(dist.isend, send, right), dist.P2POp(dist.irecv, recv, left)])
    for r in reqs:
        r.wait()


def ring_reduce_scatter(x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    """Returns (buffer with this rank's reduced chunk filled in, owned chunk index, bytes this rank sent)."""
    rank, n = dist.get_rank(), dist.get_world_size()
    chunks = list(x.clone().chunk(n))
    sent = 0
    buf = torch.empty_like(chunks[0])
    for k in range(n - 1):
        s, r = (rank - k) % n, (rank - k - 1) % n
        _exchange(chunks[s].contiguous(), buf, rank, n)
        chunks[r] += buf
        sent += chunks[s].numel() * chunks[s].element_size()
    return torch.cat(chunks), (rank + 1) % n, sent


def ring_all_gather(x: torch.Tensor, owned: int) -> tuple[torch.Tensor, int]:
    """x holds a correct chunk `owned`; circulate until every rank has every chunk."""
    rank, n = dist.get_rank(), dist.get_world_size()
    chunks = list(x.clone().chunk(n))
    sent = 0
    buf = torch.empty_like(chunks[0])
    for k in range(n - 1):
        s, r = (owned - k) % n, (owned - k - 1) % n
        _exchange(chunks[s].contiguous(), buf, rank, n)
        chunks[r] = buf.clone()
        sent += chunks[s].numel() * chunks[s].element_size()
    return torch.cat(chunks), sent


def ring_all_reduce(x: torch.Tensor) -> tuple[torch.Tensor, int]:
    """AllReduce = ReduceScatter + AllGather. Returns (sum over ranks, bytes this rank sent)."""
    partial, owned, sent_rs = ring_reduce_scatter(x)
    out, sent_ag = ring_all_gather(partial, owned)
    return out, sent_rs + sent_ag
