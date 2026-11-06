"""Effective data under repetition, from Muennighoff et al. (2023), "Scaling Data-Constrained Language Models".

With U unique tokens seen for E epochs (R = E - 1 repeats), the effective number of tokens is

    D' = U + U * R* * (1 - exp(-R / R*))

R* is fitted in the paper. Its value there is about 15.4. Each extra repeat is worth a bit less than the last.
"""

from __future__ import annotations

import numpy as np

R_STAR_PAPER = 15.4


def effective_tokens(unique: float, epochs: float | np.ndarray, r_star: float = R_STAR_PAPER):
    r = np.maximum(np.asarray(epochs, dtype=float) - 1, 0)
    return unique + unique * r_star * (1 - np.exp(-r / r_star))


def value_fraction(epochs, r_star: float = R_STAR_PAPER):
    """D' divided by the tokens actually seen (U * E). 1 means repeats were as good as fresh data."""
    e = np.asarray(epochs, dtype=float)
    return effective_tokens(1.0, e, r_star) / e


def interp_fresh(fresh_bytes: np.ndarray, fresh_loss: np.ndarray, loss: float) -> float:
    """Bytes of fresh data a single-epoch run needed to reach `loss` (log-linear interpolation).

    Returns nan if `loss` is lower than anything the fresh run reached.
    """
    order = np.argsort(fresh_bytes)
    x, y = np.log(fresh_bytes[order]), fresh_loss[order]
    if loss < y.min():
        return float("nan")
    if loss >= y.max():
        return float(fresh_bytes[order][0])
    for i in range(len(y) - 1):  # the fresh curve can wiggle, so take the first crossing
        if (y[i] - loss) * (y[i + 1] - loss) <= 0:
            f = (loss - y[i]) / (y[i + 1] - y[i]) if y[i + 1] != y[i] else 0.0
            return float(np.exp(x[i] + f * (x[i + 1] - x[i])))
    return float("nan")


def fit_r_star(unique: np.ndarray, epochs: np.ndarray, measured_eff: np.ndarray) -> float:
    """Least-squares R* on log effective tokens, by grid search (no scipy needed)."""
    grid = np.exp(np.linspace(np.log(0.1), np.log(1000), 4000))
    ok = np.isfinite(measured_eff)
    errs = [np.mean((np.log(effective_tokens(unique[ok], epochs[ok], r)) - np.log(measured_eff[ok])) ** 2) for r in grid]
    return float(grid[int(np.argmin(errs))])
