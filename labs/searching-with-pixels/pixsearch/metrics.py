"""Ranking metrics from the post's section "How retrieval is measured".

Every function takes a ranked list of candidate ids (best first) and the set of relevant ids
(or a dict id -> graded relevance for nDCG). The `mean_*` helpers average over queries.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence


def recall_at_k(ranking: Sequence[int], relevant: Iterable[int], k: int) -> float:
    """1.0 if any relevant candidate is in the top k, else 0.0 (the "hit" form used for Recall@k
    on Flickr30K and COCO, where each query has one target)."""
    rel = set(relevant)
    return float(any(c in rel for c in ranking[:k]))


def precision_at_1(ranking: Sequence[int], relevant: Iterable[int]) -> float:
    return recall_at_k(ranking, relevant, 1)


def reciprocal_rank(ranking: Sequence[int], relevant: Iterable[int]) -> float:
    """1 / rank of the first relevant candidate (rank counts from 1); 0 if none is ranked."""
    rel = set(relevant)
    for r, c in enumerate(ranking, start=1):
        if c in rel:
            return 1.0 / r
    return 0.0


def dcg_at_k(gains: Sequence[float], k: int) -> float:
    """DCG@k = sum_r (2^rel_r - 1) / log2(r + 1), with r counted from 1."""
    return sum((2.0 ** g - 1.0) / math.log2(r + 1) for r, g in enumerate(gains[:k], start=1))


def ndcg_at_k(ranking: Sequence[int], relevance: Mapping[int, float] | Iterable[int], k: int) -> float:
    """nDCG@k = DCG@k / IDCG@k. `relevance` is a dict of graded labels, or a set (binary labels)."""
    if not isinstance(relevance, Mapping):
        relevance = {c: 1.0 for c in relevance}
    gains = [relevance.get(c, 0.0) for c in ranking[:k]]
    ideal = sorted(relevance.values(), reverse=True)
    idcg = dcg_at_k(ideal, k)
    return dcg_at_k(gains, k) / idcg if idcg > 0 else 0.0


def evaluate(rankings: Sequence[Sequence[int]], relevant: Sequence[Iterable[int]],
             ks: Sequence[int] = (1, 5, 10, 50)) -> dict[str, float]:
    """Average Recall@k, P@1, MRR and nDCG@5 / nDCG@10 over queries, in percent."""
    n = len(rankings)
    rel = [set(r) for r in relevant]
    out: dict[str, float] = {}
    for k in ks:
        out[f"R@{k}"] = 100 * sum(recall_at_k(r, g, k) for r, g in zip(rankings, rel)) / n
    out["P@1"] = 100 * sum(precision_at_1(r, g) for r, g in zip(rankings, rel)) / n
    out["MRR"] = 100 * sum(reciprocal_rank(r, g) for r, g in zip(rankings, rel)) / n
    for k in (5, 10):
        out[f"nDCG@{k}"] = 100 * sum(ndcg_at_k(r, g, k) for r, g in zip(rankings, rel)) / n
    return out
