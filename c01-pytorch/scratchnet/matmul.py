"""Matrix multiplication, from three loops to one kernel call.

The loop versions are the definition. They are hundreds of thousands of times slower
than `torch.mm`, which is why nobody ships them.
"""

from __future__ import annotations

import torch


def mm_loops(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """[M, K] @ [K, N] -> [M, N] with explicit loops: out[i, j] = sum_k a[i, k] * b[k, j]."""
    m, k = a.shape
    k2, n = b.shape
    if k != k2:
        raise ValueError(f"inner dims differ: {tuple(a.shape)} @ {tuple(b.shape)}")
    out = torch.zeros(m, n, dtype=a.dtype)
    for i in range(m):
        for j in range(n):
            acc = 0.0
            for p in range(k):
                acc += a[i, p].item() * b[p, j].item()
            out[i, j] = acc
    return out


def mm_rows(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """[M, K] @ [K, N] -> [M, N], one row at a time: row i of the output is a[i] @ b."""
    return torch.stack([(a[i, :, None] * b).sum(dim=0) for i in range(a.shape[0])])


def bmm_loop(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """[B, M, K] @ [B, K, N] -> [B, M, N]: an independent matmul per batch element."""
    return torch.stack([a[i] @ b[i] for i in range(a.shape[0])])


def mm_flops(m: int, k: int, n: int) -> int:
    """One multiply and one add per (i, j, k) triple."""
    return 2 * m * k * n
