"""Allocating a sample budget across problems: water-filling on the coverage curve.

Problem i with success rate p_i and b_i samples is solved with chance 1 - (1 - p_i)^b_i.
That is concave in b_i, so giving each unit to the problem with the largest marginal gain is optimal.
"""

from __future__ import annotations

import heapq
import math

import numpy as np


def marginal_gain(p: float, b: int) -> float:
    """Gain in solve chance from sample b + 1: (1 - p)^b * p."""
    return (1.0 - p) ** b * p


def water_fill(ps, budget: int, cap: int | None = None) -> np.ndarray:
    """Greedy integer allocation. Optimal because every gain sequence is decreasing."""
    alloc = np.zeros(len(ps), dtype=int)
    heap = [(-marginal_gain(p, 0), i) for i, p in enumerate(ps) if p > 0]
    heapq.heapify(heap)
    for _ in range(budget):
        if not heap:
            break
        g, i = heapq.heappop(heap)
        alloc[i] += 1
        if cap is None or alloc[i] < cap:
            heapq.heappush(heap, (-marginal_gain(ps[i], alloc[i]), i))
    return alloc


def last_gain(ps, alloc) -> float:
    """The marginal gain of the last unit spent: the shadow price lambda (up to integer rounding)."""
    return min(marginal_gain(p, b - 1) for p, b in zip(ps, alloc) if b > 0)


def closed_form(p: float, lam: float, exact: bool = False) -> float:
    """Continuous optimum. Approximate form: (1/p) ln(p / lam) if p > lam, else 0.

    exact=True solves (1-p)^b ln(1/(1-p)) = lam without the small-p step ln(1/(1-p)) ~ p.
    """
    if exact:
        r = math.log(1.0 / (1.0 - p))
        return max(0.0, math.log(r / lam) / r)
    return max(0.0, math.log(p / lam) / p) if p > lam else 0.0


def expected_solved(ps, alloc) -> float:
    return float(sum(1.0 - (1.0 - p) ** b for p, b in zip(ps, alloc)))


def uniform(n: int, budget: int) -> np.ndarray:
    alloc = np.full(n, budget // n, dtype=int)
    alloc[: budget - alloc.sum()] += 1
    return alloc


def stop_when_solved(ps, budget: int, rng: np.random.Generator, trials: int = 200) -> float:
    """Sequential policy with a perfect verifier and no knowledge of p.

    Each sample goes to the unsolved problem with the fewest failures so far (round-robin over
    unsolved problems). A solved problem gets nothing more. Returns the mean number solved.
    """
    ps = np.asarray(ps, float)
    solved_total = 0.0
    for _ in range(trials):
        solved = np.zeros(len(ps), bool)
        tries = np.zeros(len(ps), int)
        for _ in range(budget):
            open_ = np.flatnonzero(~solved)
            if open_.size == 0:
                break
            i = open_[np.argmin(tries[open_])]
            tries[i] += 1
            solved[i] = rng.random() < ps[i]
        solved_total += solved.sum()
    return solved_total / trials
