"""The naive mixing rules and the accounting that exposes them (post sections 4 and 5).

Everything here is arithmetic on source sizes and lengths. No model is involved, which is the point: the
failure of each rule can be read off before a single training step.
"""

from __future__ import annotations

import math
from typing import Sequence


def proportional(sizes: Sequence[float]) -> list[float]:
    """p_i = |D_i| / sum_j |D_j|."""
    total = sum(sizes)
    return [s / total for s in sizes]


def equal(sizes: Sequence[float]) -> list[float]:
    """p_i = 1 / K."""
    return [1.0 / len(sizes)] * len(sizes)


def temperature(sizes: Sequence[float], alpha: float) -> list[float]:
    """p_i proportional to |D_i| ** alpha. alpha = 1 is proportional, alpha = 0 is equal."""
    raw = [s**alpha for s in sizes]
    total = sum(raw)
    return [r / total for r in raw]


def capped(sizes: Sequence[float], cap: float) -> list[float]:
    """p_i proportional to min(|D_i|, cap): T5 / FLAN examples-proportional mixing with a limit."""
    raw = [min(s, cap) for s in sizes]
    total = sum(raw)
    return [r / total for r in raw]


def n_eff(p: Sequence[float]) -> float:
    """exp(H(p)): how many sources the mixture is really training on. K when uniform, 1 when one source owns it."""
    return math.exp(-sum(x * math.log(x) for x in p if x > 0))


def epochs(p: Sequence[float], sizes: Sequence[float], budget: float) -> list[float]:
    """Times each source is seen when `budget` examples are drawn: budget * p_i / |D_i|."""
    return [budget * pi / s for pi, s in zip(p, sizes)]


def token_weights(p: Sequence[float], mean_len: Sequence[float]) -> list[float]:
    """Gradient share under a token-mean loss: lambda_i = p_i * l_i / sum_j p_j * l_j."""
    raw = [pi * li for pi, li in zip(p, mean_len)]
    total = sum(raw)
    return [r / total for r in raw]


def example_weights_for_token_target(target: Sequence[float], mean_len: Sequence[float]) -> list[float]:
    """Invert token_weights: the example-level p that gives a wanted token-level share."""
    raw = [t / li for t, li in zip(target, mean_len)]
    total = sum(raw)
    return [r / total for r in raw]


def log_spaced_sizes(k: int = 24, smallest: float = 700, largest: float = 2_000_000) -> list[int]:
    """K source sizes spread evenly in log space: a power-law-like spread from a hand-written set to a scraped one."""
    ratio = (largest / smallest) ** (1 / (k - 1))
    return [round(smallest * ratio**i) for i in range(k)]
