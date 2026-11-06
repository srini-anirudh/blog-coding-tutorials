"""Matrix multiplication, from three loops to one kernel call.

The loop versions are the definition. Per flop they are about a million times
slower than `torch.mm` (results/matmul.txt), which is why nobody ships them.
"""

from __future__ import annotations

import torch

from .broadcasting import broadcast_shape


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


def matmul_manual(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """[..., M, K] @ [..., K, N] -> [..., M, N], for inputs of rank 2 or more.

    This is what torch.matmul adds on top of torch.bmm: the leading dims broadcast.
    The last two dims never do. They follow the matmul rule instead.
    """
    batch = broadcast_shape(tuple(a.shape[:-2]), tuple(b.shape[:-2]))
    m, k = a.shape[-2:]
    k2, n = b.shape[-2:]
    a_flat = a.expand(*batch, m, k).reshape(-1, m, k)  # [prod(batch), M, K]
    b_flat = b.expand(*batch, k2, n).reshape(-1, k2, n)  # [prod(batch), K, N]
    return bmm_loop(a_flat, b_flat).reshape(*batch, m, n)


def mm_flops(m: int, k: int, n: int) -> int:
    """One multiply and one add per (i, j, k) triple."""
    return 2 * m * k * n
