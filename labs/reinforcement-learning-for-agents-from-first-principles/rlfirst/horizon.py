"""Where the signal goes: the arithmetic of group-relative advantages over long horizons.

A group of G rollouts on one task, binary reward, success probability p. The group is useless when every rollout
agrees, so

    P_informative(p, G) = 1 - p^G - (1 - p)^G .

With H independent decisions, each right with probability q, p = q^H. For small p, P_informative ~ G q^H.

The same model gives the fifty-percent horizon H_50 = ln 0.5 / ln q, about 0.69 / (1 - q) for q near 1. A policy
that catches an error with probability r before it becomes terminal fails a step with probability (1 - q)(1 - r)
instead, which stretches H_50 without making any single step more reliable.

Every closed form here has a Monte Carlo twin, so the formulas are checked, not assumed.
"""

from __future__ import annotations

import math

import numpy as np


def p_informative(p: float | np.ndarray, G: int) -> float | np.ndarray:
    return 1.0 - p ** G - (1.0 - p) ** G


def p_success(q: float, H: int) -> float:
    return q ** H


def mc_informative(p: float, G: int, n_groups: int, rng: np.random.Generator) -> float:
    """Fraction of sampled groups whose binary rewards are not all equal."""
    r = rng.random((n_groups, G)) < p
    s = r.sum(1)
    return float(((s > 0) & (s < G)).mean())


def mc_informative_horizon(q: float, H: int, G: int, n_groups: int, rng: np.random.Generator) -> float:
    """Same, but each rollout is H independent steps, each right w.p. q. Success = all H right."""
    ok = (rng.random((n_groups, G, H)) < q).all(-1)
    s = ok.sum(1)
    return float(((s > 0) & (s < G)).mean())


def group_size_needed(q: float, H: int, target: float) -> int:
    """Smallest G with P_informative(q^H, G) >= target, for p = q^H <= 1/2. Grows like q^-H."""
    p = q ** H
    hi = 2
    while p_informative(p, hi) < target:              # P_informative rises with G while p <= 1/2
        hi *= 2
    lo = hi // 2
    while lo < hi:
        mid = (lo + hi) // 2
        lo, hi = (lo, mid) if p_informative(p, mid) >= target else (mid + 1, hi)
    return lo


def graded_rewards(q: float, H: int, size: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    """Partial credit: the fraction of the H steps completed before the first error (1.0 = success)."""
    right = rng.random(size + (H,)) < q
    first_wrong = np.where(right.all(-1), H, np.argmin(right, -1))
    return first_wrong / H


def mc_informative_graded(q: float, H: int, G: int, n_groups: int, rng: np.random.Generator) -> float:
    r = graded_rewards(q, H, (n_groups, G), rng)
    return float((r.std(1) > 0).mean())


def h50(q: float) -> float:
    return math.log(0.5) / math.log(q)


def h50_approx(q: float) -> float:
    return math.log(2) / (1.0 - q)


def mc_success_with_recovery(q: float, r: float, H: int, n: int, rng: np.random.Generator) -> float:
    """H steps. Each step errs w.p. 1 - q; an error is caught w.p. r (the step is retried), else the task fails.

    Retries can err again, so a step is eventually passed w.p. q / (1 - (1-q) r). No step budget.
    """
    alive = np.ones(n, dtype=bool)
    for _ in range(H):
        pending = alive.copy()
        while pending.any():
            err = rng.random(n) < (1 - q)
            caught = rng.random(n) < r
            alive &= ~(pending & err & ~caught)              # uncaught error: the episode is over
            pending &= err & caught                          # caught error: try the step again
    return float(alive.mean())


def h50_with_recovery(q: float, r: float) -> float:
    """Closed form for the model above: per-step pass probability q_eff = q / (1 - (1 - q) r)."""
    return h50(q / (1 - (1 - q) * r))
