"""Index a corpus, rank it for each query, grade the rankings, and time the query side."""

from __future__ import annotations

import time

import torch

from . import pages as P
from .metrics import evaluate
from .models import BiEncoder, CrossEncoder, LateInteraction


@torch.no_grad()
def index_pages(model, images: torch.Tensor, chunk: int = 512) -> torch.Tensor:
    """Offline step: encode every page once. Bi: N x dim. Late: N x 16 x dim."""
    return torch.cat([model.encode_pages(images[i:i + chunk]) for i in range(0, len(images), chunk)])


@torch.no_grad()
def score_all(model, tokens: torch.Tensor, index: torch.Tensor, chunk: int = 64) -> torch.Tensor:
    """Q x N scores of every query against the stored index."""
    out = []
    for i in range(0, len(tokens), chunk):
        q = model.encode_queries(tokens[i:i + chunk])
        out.append(model.score(q, index))
    return torch.cat(out)


@torch.no_grad()
def cross_score_all(model: CrossEncoder, tokens: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
    """A cross-encoder must run one forward pass per (query, page) pair."""
    return torch.stack([model.score_pairs(t, images) for t in tokens])


def rankings(scores: torch.Tensor, depth: int = 100) -> list[list[int]]:
    return scores.topk(min(depth, scores.shape[1]), dim=1).indices.tolist()


def twin_win_rate(scores: torch.Tensor, queries: P.Queries) -> float:
    """Percent of queries whose source page outscores its twin (pages p and p ^ 1 are twins)."""
    src = queries.source
    s = scores[torch.arange(len(src)), src]
    t = scores[torch.arange(len(src)), src ^ 1]
    return 100 * (s > t).float().mean().item()


def evaluate_scores(scores: torch.Tensor, queries: P.Queries, rel: list[set[int]], twins: bool = True) -> dict:
    out = evaluate(rankings(scores), rel)
    if twins:
        out["beats_twin"] = twin_win_rate(scores, queries)
    return {k: round(v, 2) for k, v in out.items()}


def time_queries(fn, reps: int = 3) -> float:
    """Best-of-`reps` wall-clock seconds for fn()."""
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def stored_floats_per_page(model) -> int:
    if isinstance(model, LateInteraction):
        return P.N_PATCHES * model.dim
    if isinstance(model, BiEncoder):
        return model.dim
    return 0
