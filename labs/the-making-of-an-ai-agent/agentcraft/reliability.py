"""Reliability is not accuracy: pass@k against pass^k (tau-bench).

For a task with true success probability p and k independent trials:
    pass@k = 1 - (1 - p)^k     at least one trial succeeds
    pass^k = p^k               every trial succeeds
From n trials with c successes, the unbiased estimators are
    pass@k = 1 - C(n - c, k) / C(n, k)        pass^k = C(c, k) / C(n, k)
Averaged over tasks, pass^1 = pass@1 = mean success. For k > 1 the two depend on how success is spread over tasks,
not only on its mean.
"""

from __future__ import annotations

import random
from math import comb


def pass_at_k(n: int, c: int, k: int) -> float:
    return 1.0 - comb(n - c, k) / comb(n, k)


def pass_hat_k(n: int, c: int, k: int) -> float:
    return comb(c, k) / comb(n, k)


def exact_pass_hat_k(ps: list[float], k: int) -> float:
    return sum(p ** k for p in ps) / len(ps)


def exact_pass_at_k(ps: list[float], k: int) -> float:
    return sum(1 - (1 - p) ** k for p in ps) / len(ps)


def estimate(ps: list[float], n: int, k: int, seed: int = 0) -> tuple[float, float]:
    """Simulate n trials per task, then apply the unbiased estimators. Returns (pass@k, pass^k)."""
    rng = random.Random(seed)
    cs = [sum(rng.random() < p for _ in range(n)) for p in ps]
    return (sum(pass_at_k(n, c, k) for c in cs) / len(cs), sum(pass_hat_k(n, c, k) for c in cs) / len(cs))
