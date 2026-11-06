"""Signal, noise and decision accuracy, after Heineman et al. (2025) and DataDecide (Magnusson et al., 2025).

For one metric:
- signal: how far apart different recipes score. Here: the standard deviation of the final score across recipes.
- noise: how much one run's score moves between its last few checkpoints. Here: the standard deviation over
  the last k checkpoints, averaged over runs.
- decision accuracy: the share of recipe pairs that the small-scale metric orders the same way as the
  large-scale target does.
"""

from __future__ import annotations

import itertools

import numpy as np


def signal(final_scores: np.ndarray) -> float:
    return float(np.std(final_scores))


def noise(last_k_scores: np.ndarray, detrend: bool = False) -> float:
    """(n_runs, k) -> mean over runs of the std across that run's last k checkpoints.

    With detrend=True a straight line is fitted to each run first and the std is taken of what is left.
    Use it when the checkpoints are still on a falling learning-rate schedule, so the score drifts.
    """
    x = np.asarray(last_k_scores, dtype=float)
    if detrend:
        t = np.arange(x.shape[1])
        x = np.stack([r - np.polyval(np.polyfit(t, r, 1), t) for r in x])
        return float(np.mean(np.sqrt((x**2).sum(1) / max(x.shape[1] - 2, 1))))
    return float(np.mean(np.std(x, axis=1)))


def snr(final_scores: np.ndarray, last_k_scores: np.ndarray, detrend: bool = False) -> float:
    return signal(final_scores) / max(noise(last_k_scores, detrend), 1e-12)


def decision_accuracy(small: np.ndarray, large: np.ndarray) -> float:
    """Pairwise ranking agreement. Lower is better for both arrays (losses). Ties in `large` are skipped."""
    agree, total = 0, 0
    for i, j in itertools.combinations(range(len(small)), 2):
        if large[i] == large[j]:
            continue
        total += 1
        agree += (small[i] < small[j]) == (large[i] < large[j])
    return agree / max(total, 1)


def rank_flip_rate(last_k_a: np.ndarray, last_k_b: np.ndarray) -> float:
    """Two runs, k checkpoints each. Share of checkpoints where the order differs from the order of the means."""
    sign = np.sign(last_k_a.mean() - last_k_b.mean())
    return float(np.mean(np.sign(last_k_a - last_k_b) != sign))
