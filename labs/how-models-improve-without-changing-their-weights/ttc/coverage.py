"""Coverage: why repeated sampling has a scaling law.

One problem with success rate p has coverage 1 - (1 - p)^k. That is a sigmoid in log k.
A benchmark is a mix of problems. If p ~ Beta(a, b), the uncovered fraction E[(1 - p)^k]
decays like k^(-a), a straight line on a log-log plot.
"""

from __future__ import annotations

import math

import numpy as np


def coverage_one(p: float, k):
    """Chance that at least one of k independent samples is correct."""
    return 1.0 - (1.0 - p) ** np.asarray(k, dtype=float)


def half_coverage_k(p: float) -> float:
    """Number of samples at which one problem reaches 50% coverage. Close to 0.69 / p."""
    return math.log(2.0) / math.log(1.0 / (1.0 - p))


def log_beta(a: float, b: float) -> float:
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def beta_uncovered(a: float, b: float, k: float) -> float:
    """Exact E[(1 - p)^k] for p ~ Beta(a, b): B(a, b + k) / B(a, b)."""
    return math.exp(log_beta(a, b + k) - log_beta(a, b))


def beta_uncovered_tail(a: float, b: float, k: float) -> float:
    """Large-k form of the same quantity: Gamma(a + b) / Gamma(b) * k^(-a)."""
    return math.exp(math.lgamma(a + b) - math.lgamma(b)) * k ** (-a)


def loglog_slope(ks, values) -> float:
    """Least-squares slope of log(values) against log(ks)."""
    return float(np.polyfit(np.log(np.asarray(ks, float)), np.log(np.asarray(values, float)), 1)[0])


def sample_population(a: float, b: float, n: int, pi0: float, rng: np.random.Generator) -> np.ndarray:
    """Draw n per-problem success rates. A fraction pi0 of problems is hopeless (p = 0)."""
    p = rng.beta(a, b, size=n)
    p[rng.random(n) < pi0] = 0.0
    return p


def simulate_coverage(p: np.ndarray, ks, rng: np.random.Generator) -> np.ndarray:
    """Monte Carlo coverage: sample until the first success for every problem, then count."""
    first = np.full(p.shape, np.inf)
    ok = p > 0
    first[ok] = rng.geometric(p[ok])  # index of the first correct sample, 1-based
    return np.array([(first <= k).mean() for k in ks])


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k from n samples with c correct (Chen et al., 2021): 1 - C(n-c, k) / C(n, k)."""
    if n - c < k:
        return 1.0
    return 1.0 - float(np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def fit_beta_binomial(successes, n: int, grid=None) -> tuple[float, float]:
    """Maximum-likelihood (a, b) of a Beta prior from per-problem success counts out of n.

    Grid search on log-spaced values, so no optimiser library is needed.
    """
    counts = np.bincount(np.asarray(successes, int), minlength=n + 1)  # problems with s successes
    s = np.arange(n + 1, dtype=float)
    grid = np.exp(np.linspace(np.log(0.02), np.log(50.0), 120)) if grid is None else grid
    best, best_ab = -np.inf, (1.0, 1.0)
    lg = np.vectorize(math.lgamma)
    for a in grid:
        la = lg(s + a)
        for b in grid:
            per = la + lg(n - s + b) - math.lgamma(n + a + b) - log_beta(a, b)
            ll = float((counts * per).sum())
            if ll > best:
                best, best_ab = ll, (float(a), float(b))
    return best_ab
