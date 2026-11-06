"""Who can I see, and who can I reach after L layers.

A mask is a directed graph: edge i -> j when position i may read position j. One layer moves
information along one edge. L layers reach what the L-th boolean power of the mask reaches
(with self-loops, which the residual stream provides). Everything here is a bool (N, N) tensor.
"""

from __future__ import annotations

import math

import torch

from .common import attention

# ---------- 1D patterns (True = may attend) ----------


def causal(n: int) -> torch.Tensor:
    return torch.ones(n, n, dtype=torch.bool).tril()


def sliding_window(n: int, w: int, is_causal: bool = True) -> torch.Tensor:
    """Each query sees the w most recent keys, itself included (or |i - j| < w if not causal)."""
    i, j = torch.arange(n)[:, None], torch.arange(n)[None]
    band = (i - j).abs() < w
    return band & (j <= i) if is_causal else band


def document_mask(doc_ids: torch.Tensor, is_causal: bool = True) -> torch.Tensor:
    """Packed sequences: attend only inside your own document."""
    same = doc_ids[:, None] == doc_ids[None]
    return same & causal(len(doc_ids)) if is_causal else same


def block_diagonal(n: int, block: int) -> torch.Tensor:
    """Attend only inside fixed blocks: no edge ever crosses a block boundary."""
    i, j = torch.arange(n)[:, None], torch.arange(n)[None]
    return i // block == j // block


def strided(n: int, stride: int) -> torch.Tensor:
    """Sparse Transformer style, bidirectional here: your own block, plus every stride-th column."""
    i, j = torch.arange(n)[:, None], torch.arange(n)[None]
    return (i // stride == j // stride) | ((j % stride) == stride - 1)


def local_global(n: int, w: int, n_global: int) -> torch.Tensor:
    """Longformer style: a band, plus a few positions that see and are seen by everyone."""
    m = sliding_window(n, w, is_causal=False)
    m[:n_global, :] = True
    m[:, :n_global] = True
    return m


def local_random(n: int, w: int, r: int, seed: int = 0) -> torch.Tensor:
    """Big Bird style without the global tokens: a band, plus r random keys per query."""
    g = torch.Generator().manual_seed(seed)
    m = sliding_window(n, w, is_causal=False)
    m[torch.arange(n)[:, None], torch.randint(n, (n, r), generator=g)] = True
    return m


# ---------- 2D patterns over an H x W grid, flattened row-major ----------


def grid_windows(h: int, w: int, win: int, shift: int = 0) -> torch.Tensor:
    """Swin-style windows of win x win patches. shift rolls the partition (cyclically, as Swin does)."""
    r, c = torch.arange(h * w) // w, torch.arange(h * w) % w
    wid = ((r + shift) % h // win) * (w // win) + ((c + shift) % w // win)
    return wid[:, None] == wid[None]


def axial(h: int, w: int) -> tuple[torch.Tensor, torch.Tensor]:
    r, c = torch.arange(h * w) // w, torch.arange(h * w) % w
    return r[:, None] == r[None], c[:, None] == c[None]


# ---------- reachability ----------


def reach(layers: list[torch.Tensor]) -> torch.Tensor:
    """Which inputs each output can depend on after running these masks in order (residual included)."""
    n = layers[0].size(0)
    r = torch.eye(n, dtype=torch.bool)
    for m in layers:
        m = m | torch.eye(n, dtype=torch.bool)
        r = (m.float() @ r.float()) > 0  # output i reads j in this layer, and j already depends on r[j]
    return r


def hops_to_full(pattern, max_layers: int = 64) -> int | None:
    """Layers needed until every output depends on every allowed input. pattern(layer) -> mask.
    Fully connected means all pairs, or all j <= i if the pattern is causal. None if never."""
    first = pattern(0)
    target = causal(first.size(0)) if not first.triu(1).any() else torch.ones_like(first)
    layers = []
    for depth in range(1, max_layers + 1):
        layers.append(pattern(depth - 1))
        if (reach(layers) >= target).all():
            return depth
    return None


def farthest_reach(n: int, w: int, depth: int) -> int:
    """How far back the last position can see after `depth` causal sliding-window layers."""
    r = reach([sliding_window(n, w)] * depth)
    return n - 1 - int(r[-1].nonzero().min())


# ---------- the same thing, measured on real attention layers with autograd ----------


def gradient_reach(n: int, w: int, depth: int, d: int = 16, seed: int = 0) -> int:
    """Stack `depth` random attention layers with a causal window and residuals, then ask autograd
    which inputs the last output actually depends on. Returns the farthest nonzero distance."""
    g = torch.Generator().manual_seed(seed)
    weights = [[torch.randn(d, d, generator=g, dtype=torch.float64) / math.sqrt(d) for _ in range(3)] for _ in range(depth)]
    x = torch.randn(n, d, generator=g, dtype=torch.float64, requires_grad=True)
    h, m = x, sliding_window(n, w)
    for wq, wk, wv in weights:
        h = h + attention(h @ wq, h @ wk, h @ wv, mask=m)[0]
    h[-1].sum().backward()
    nonzero = (x.grad.abs().sum(-1) > 0).nonzero()
    return n - 1 - int(nonzero.min())


def packing_leak(doc_lengths: list[int], d: int = 16, seed: int = 0) -> dict:
    """Pack documents into one sequence, change document 0, and see if later documents notice."""
    g = torch.Generator().manual_seed(seed)
    n = sum(doc_lengths)
    doc_ids = torch.repeat_interleave(torch.arange(len(doc_lengths)), torch.tensor(doc_lengths))
    wq, wk, wv = (torch.randn(d, d, generator=g) / math.sqrt(d) for _ in range(3))
    x = torch.randn(n, d, generator=g)
    x2 = x.clone()
    x2[: doc_lengths[0]] = torch.randn(doc_lengths[0], d, generator=g)
    later = doc_ids > 0
    out = {}
    for name, m in [("causal only", causal(n)), ("causal AND same document", document_mask(doc_ids))]:
        a = attention(x @ wq, x @ wk, x @ wv, mask=m)[0]
        b = attention(x2 @ wq, x2 @ wk, x2 @ wv, mask=m)[0]
        out[name] = (a[later] - b[later]).abs().max().item()
    return out


def padding_and_causal(lengths: list[int], n: int, d: int = 16, seed: int = 0) -> float:
    """Padding (column mask) AND causal (triangle). Real tokens must match an unpadded run exactly."""
    g = torch.Generator().manual_seed(seed)
    wq, wk, wv = (torch.randn(d, d, generator=g) / math.sqrt(d) for _ in range(3))
    x = torch.randn(len(lengths), n, d, generator=g)
    valid = torch.arange(n)[None] < torch.tensor(lengths)[:, None]  # (B, N)
    m = causal(n)[None] & valid[:, None, :]  # (B, N, N): AND of the two
    padded = attention(x @ wq, x @ wk, x @ wv, mask=m)[0]
    worst = 0.0
    for b, length in enumerate(lengths):
        h = x[b, :length]
        alone = attention(h @ wq, h @ wk, h @ wv, mask=causal(length))[0]
        worst = max(worst, (padded[b, :length] - alone).abs().max().item())
    return worst
