"""A grader with error rates alpha (passes a wrong solution) and beta (fails a correct one), and what it costs."""

from __future__ import annotations

import numpy as np

Z_95, Z_80 = 1.959964, 0.841621   # two-sided 5% test, 80% power


def measured(pi, alpha: float, beta: float):
    """What the leaderboard prints: pi(1 - beta) + (1 - pi) alpha."""
    pi = np.asarray(pi, dtype=float)
    return pi * (1 - beta) + (1 - pi) * alpha


def grade(true_pass: np.ndarray, alpha: float, beta: float, rng) -> np.ndarray:
    """Pass each true success with probability 1-beta and each true failure with probability alpha."""
    u = rng.random(true_pass.shape)
    return np.where(true_pass, u >= beta, u < alpha)


def power(n: int, pi_a: float, pi_b: float, alpha: float, beta: float, rng, sims: int = 20000) -> float:
    """Chance that a two-proportion z-test at 5% separates two agents from n graded runs each.

    Each run draws a true outcome, then passes it through the noisy grader, so this checks the formula rather than
    assuming it. Counts are drawn as binomials, which is the same process as grading runs one by one.
    """
    def graded_rate(pi):
        true = rng.binomial(n, pi, sims)
        return (rng.binomial(true, 1 - beta) + rng.binomial(n - true, alpha)) / n

    a, b = graded_rate(pi_a), graded_rate(pi_b)
    pool = (a + b) / 2
    se = np.sqrt(np.maximum(2 * pool * (1 - pool) / n, 1e-12))
    return float(np.mean(np.abs(b - a) / se > Z_95))


def runs_needed_sim(pi_a, pi_b, alpha, beta, rng, target: float = 0.8, sims: int = 20000) -> int:
    """Smallest n with simulated power >= target, by bisection over n."""
    lo, hi = 2, 8
    while power(hi, pi_a, pi_b, alpha, beta, rng, sims) < target:
        lo, hi = hi, hi * 2
    while hi - lo > max(1, lo // 50):
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if power(mid, pi_a, pi_b, alpha, beta, rng, sims) < target else (lo, mid)
    return hi


def runs_needed_normal(pi_a, pi_b, alpha, beta) -> float:
    """The normal-approximation sample size, using the measured (compressed) rates."""
    ma, mb = measured(pi_a, alpha, beta), measured(pi_b, alpha, beta)
    return float((Z_95 + Z_80) ** 2 * (ma * (1 - ma) + mb * (1 - mb)) / (mb - ma) ** 2)
