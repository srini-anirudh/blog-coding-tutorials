"""Token reduction after the fact, from "Pruning and merging after the fact".

Every policy takes the visual tokens of one image on an h x w grid and returns k tokens as a `Reduced`:
the kept (or merged) features, the grid index each output token takes its position from, and how many
original tokens each output token stands for.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class Reduced:
    features: torch.Tensor   # k x d
    index: torch.Tensor      # k, grid index (row-major) whose position the token keeps
    size: torch.Tensor       # k, original tokens merged into this one


def _keep(features: torch.Tensor, idx: torch.Tensor) -> Reduced:
    idx = idx.sort().values
    return Reduced(features[idx], idx, torch.ones(len(idx), dtype=torch.long, device=features.device))


def keep_all(features, h, w, k=None, **_) -> Reduced:
    return _keep(features, torch.arange(h * w))


def random_keep(features, h, w, k, seed: int = 0, **_) -> Reduced:
    g = torch.Generator().manual_seed(seed)
    return _keep(features, torch.randperm(h * w, generator=g)[:k])


def uniform_grid(features, h, w, k, **_) -> Reduced:
    """An evenly spaced sub-grid of about sqrt(k) x sqrt(k) tokens (exact when h, w are multiples)."""
    rows = max(1, round(math.sqrt(k * h / w)))
    cols = max(1, k // rows)
    r = torch.linspace(0, h - 1, rows).round().long()
    c = torch.linspace(0, w - 1, cols).round().long()
    return _keep(features, (r[:, None] * w + c[None, :]).flatten())


def attention_rank(features, h, w, k, scores: torch.Tensor, **_) -> Reduced:
    """FastV's rule: keep the k tokens that receive the most attention."""
    return _keep(features, scores.topk(k).indices)


def diversity(features, h, w, k, **_) -> Reduced:
    """Max-min diversity (DivPrune-style farthest-point selection on cosine distance).

    Start from the token with the largest norm, then repeatedly add the token whose closest kept token is
    farthest away.
    """
    x = F.normalize(features.float(), dim=-1)
    chosen = [int(features.float().norm(dim=-1).argmax())]
    nearest = 1 - x @ x[chosen[0]]
    for _ in range(k - 1):
        nearest[chosen] = -1.0
        nxt = int(nearest.argmax())
        chosen.append(nxt)
        nearest = torch.minimum(nearest, 1 - x @ x[nxt])
    return _keep(features, torch.tensor(chosen))


def avg_pool(features, h, w, k, **_) -> Reduced:
    """Mean of each s x s block, with s chosen so that (h/s) * (w/s) = k. Position: the block centre."""
    s = int(round(math.sqrt(h * w / k)))
    if h % s or w % s:
        raise ValueError(f"grid {h}x{w} is not divisible into {s}x{s} blocks")
    grid = features.T.reshape(1, -1, h, w).float()
    pooled = F.avg_pool2d(grid, s).flatten(2)[0].T.to(features.dtype)
    rr, cc = torch.meshgrid(torch.arange(h // s) * s + s // 2, torch.arange(w // s) * s + s // 2, indexing="ij")
    return Reduced(pooled, (rr * w + cc).flatten().to(features.device),
                   torch.full((len(pooled),), s * s, device=features.device))


# ---------------------------------------------------------------- Token Merging (ToMe)

def bipartite_merge(x: torch.Tensor, size: torch.Tensor, index: torch.Tensor, r: int,
                    keys: torch.Tensor | None = None):
    """One round of ToMe's bipartite soft matching.

    Tokens alternate into sets A (even positions) and B (odd). Each A token finds its most similar B token by
    cosine similarity of `keys` (default: the features). The r A tokens with the strongest links are averaged
    into their B partner, weighted by size. Returns merged x, size and index (B tokens keep their own index).
    """
    keys = x if keys is None else keys
    k = F.normalize(keys.float(), dim=-1)
    a, b = k[0::2], k[1::2]
    r = min(r, len(a))
    sim = a @ b.T
    best_val, best_b = sim.max(dim=-1)
    order = best_val.argsort(descending=True)
    merged_a, kept_a = order[:r], order[r:]

    xa, xb = x[0::2].float(), x[1::2].float()
    sa, sb = size[0::2].float(), size[1::2].float()
    dst = best_b[merged_a]
    xb_sum = xb * sb[:, None]
    xb_sum = xb_sum.index_add(0, dst, xa[merged_a] * sa[merged_a, None])
    sb_new = sb.index_add(0, dst, sa[merged_a])
    xb_new = xb_sum / sb_new[:, None]

    out_x = torch.cat([xa[kept_a], xb_new]).to(x.dtype)
    out_s = torch.cat([sa[kept_a], sb_new]).long()
    out_i = torch.cat([index[0::2][kept_a], index[1::2]])
    order = out_i.argsort()                    # keep raster order for the language model
    return out_x[order], out_s[order], out_i[order]


def tome(features, h, w, k, **_) -> Reduced:
    """Merge rounds until k tokens are left (each round removes at most half the tokens)."""
    dev = features.device   # size and index must live next to the features (the CUDA run needs this)
    x, size, index = features, torch.ones(h * w, dtype=torch.long, device=dev), torch.arange(h * w, device=dev)
    while len(x) > k:
        x, size, index = bipartite_merge(x, size, index, len(x) - k)
    return Reduced(x, index, size)


def proportional_attention(q: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, size: torch.Tensor):
    """softmax(q k^T / sqrt(d) + log size) v: a merged token counts as much as the patches it replaced."""
    logits = q @ keys.T / math.sqrt(q.shape[-1]) + size.float().log()
    return logits.softmax(-1) @ values


POLICIES = {"random": random_keep, "uniform": uniform_grid, "pool": avg_pool, "attention": attention_rank,
            "diversity": diversity, "tome": tome}
