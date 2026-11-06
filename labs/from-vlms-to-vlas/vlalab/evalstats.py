"""Statistics for robot trials: intervals for a success rate, power and sample size for comparing two policies."""

from __future__ import annotations

import math

import numpy as np

Z95, Z80 = 1.959963984540054, 0.8416212335729143


def normal_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def normal_interval(p: float, n: int, z: float = Z95) -> tuple[float, float]:
    half = z * math.sqrt(p * (1 - p) / n)
    return p - half, p + half


def wilson_interval(p: float, n: int, z: float = Z95) -> tuple[float, float]:
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z / (1 + z * z / n) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return centre - half, centre + half


def trials_needed(pa: float, pb: float, z_a: float = Z95, z_b: float = Z80) -> int:
    """Trials per policy for a two-sided 5% test with 80% power (the post's formula), rounded up."""
    return math.ceil((z_a + z_b) ** 2 * (pa * (1 - pa) + pb * (1 - pb)) / (pa - pb) ** 2)


def power(pa: float, pb: float, n: int, z_a: float = Z95) -> float:
    """Normal-approximation power of the two-sided two-proportion z-test with pooled variance."""
    pbar = (pa + pb) / 2
    se0 = math.sqrt(2 * pbar * (1 - pbar) / n)
    se1 = math.sqrt((pa * (1 - pa) + pb * (1 - pb)) / n)
    d = abs(pa - pb)
    return normal_cdf((d - z_a * se0) / se1) + normal_cdf((-d - z_a * se0) / se1)


def simulated_power(pa: float, pb: float, n: int, reps: int = 20000, seed: int = 0) -> float:
    """Run the experiment `reps` times: n Bernoulli trials per policy, then the pooled z-test at 5%."""
    rng = np.random.default_rng(seed)
    a = rng.binomial(n, pa, reps) / n
    b = rng.binomial(n, pb, reps) / n
    pool = (a + b) / 2
    se = np.sqrt(2 * pool * (1 - pool) / n)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(se > 0, np.abs(a - b) / se, 0.0)
    return float((z > Z95).mean())


def interval_coverage(p: float, n: int, kind: str, reps: int = 20000, seed: int = 0) -> float:
    """How often the 95% interval computed from n trials contains the true rate p."""
    rng = np.random.default_rng(seed)
    hats = rng.binomial(n, p, reps) / n
    f = wilson_interval if kind == "wilson" else normal_interval
    lo, hi = np.array([f(h, n) for h in hats]).T
    return float(((lo <= p) & (p <= hi)).mean())
