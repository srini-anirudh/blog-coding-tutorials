"""What a multi-vector index costs, and the four levers that shrink it (section "The cost of
multi-vector indexes"): vectors per page (merge or prune), dimensions, bytes per dimension, and
which pages keep multi-vector form.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .models import maxsim


def storage_bytes(pages: int, vectors: int, dims: int, bytes_per_dim: float) -> float:
    """bytes = pages x vectors per page x dimensions x bytes per dimension."""
    return pages * vectors * dims * bytes_per_dim


def _merge_one(x: torch.Tensor, target: int) -> torch.Tensor:
    """Agglomerative merge of one page's vectors (m x d) down to `target` cluster means. At each step the
    two clusters whose (unit) centroids are most similar are merged, weighted by size."""
    cent = [v for v in x]
    size = [1] * len(cent)
    while len(cent) > target:
        c = F.normalize(torch.stack(cent), dim=-1)
        sim = c @ c.T
        sim.fill_diagonal_(-float("inf"))
        i, j = divmod(int(sim.argmax()), len(cent))
        i, j = min(i, j), max(i, j)
        cent[i] = (cent[i] * size[i] + cent[j] * size[j]) / (size[i] + size[j])
        size[i] += size[j]
        del cent[j], size[j]
    return F.normalize(torch.stack(cent), dim=-1)


def merge_pool(index: torch.Tensor, factor: int) -> torch.Tensor:
    """Token pooling by merging: N x m x d -> N x ceil(m / factor) x d (query-agnostic)."""
    target = math.ceil(index.shape[1] / factor)
    if target >= index.shape[1]:
        return index
    return torch.stack([_merge_one(p, target) for p in index])


def prune(index: torch.Tensor, factor: int, mode: str = "norm", raw_norms: torch.Tensor | None = None,
          seed: int = 0) -> torch.Tensor:
    """Query-agnostic pruning: keep ceil(m / factor) vectors per page.
    mode="norm" keeps the vectors with the largest pre-normalisation norm; "random" keeps a random subset."""
    n, m, _ = index.shape
    keep = math.ceil(m / factor)
    if mode == "norm":
        assert raw_norms is not None
        idx = raw_norms.topk(keep, dim=1).indices
    elif mode == "random":
        g = torch.Generator().manual_seed(seed)
        idx = torch.stack([torch.randperm(m, generator=g)[:keep] for _ in range(n)])
    else:
        raise ValueError(mode)
    return torch.gather(index, 1, idx[..., None].expand(-1, -1, index.shape[2]))


def to_fp16(index: torch.Tensor) -> torch.Tensor:
    return index.half().float()


def to_int8(index: torch.Tensor) -> torch.Tensor:
    """Symmetric per-vector int8: x ~ round(x / s) * s, s = max|x| / 127. Returns the dequantised floats."""
    s = index.abs().amax(-1, keepdim=True).clamp_min(1e-8) / 127
    return (index / s).round().clamp(-127, 127) * s


def to_binary(index: torch.Tensor) -> torch.Tensor:
    """Keep only the sign of each dimension (1 bit), scaled to unit length."""
    return torch.sign(index) / math.sqrt(index.shape[-1])


def binary_then_rescore(q: torch.Tensor, index: torch.Tensor, k: int) -> torch.Tensor:
    """First stage with 1-bit pages (and float queries), then rescore the top k with the float index.
    Returns Q x N scores: the top k carry their float MaxSim, every other page keeps its binary score
    minus 1e6, so it stays below the rescored block in its binary order."""
    coarse = maxsim(q, to_binary(index))
    top = coarse.topk(k, dim=1).indices
    out = coarse - 1e6
    for i in range(q.shape[0]):
        out[i, top[i]] = maxsim(q[i:i + 1], index[top[i]])[0]
    return out

