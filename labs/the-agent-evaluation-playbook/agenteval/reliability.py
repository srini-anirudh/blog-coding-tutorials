"""Repeated trials and long horizons: pass@k, pass^k, cost-aware utility, hazard and H50."""

from __future__ import annotations

from math import comb, log

import numpy as np


# ---- pass@k and pass^k ---------------------------------------------------------------------------------------------
def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimate of 1 - (1-p)^k from c successes in n trials: did any of k attempts succeed?"""
    return 1.0 - comb(n - c, k) / comb(n, k)


def pass_hat_k(n: int, c: int, k: int) -> float:
    """Unbiased estimate of p^k from c successes in n trials (the tau-bench estimator): did all k succeed?"""
    return comb(c, k) / comb(n, k)


def suite_curve(successes, n: int, ks, metric=pass_hat_k) -> np.ndarray:
    """Average a per-task estimator over the suite, for each k. `successes[i]` is task i's count out of n."""
    return np.array([np.mean([metric(n, int(c), k) for c in successes]) for k in ks])


def second_order(pbar: float, var: float, k: int) -> float:
    """E[p^k] ~= pbar^k + k(k-1)/2 * pbar^(k-2) * Var(p)."""
    return pbar ** k + 0.5 * k * (k - 1) * pbar ** (k - 2) * var


def difficulty_suites(n_tasks: int, mean: float, rng) -> dict[str, np.ndarray]:
    """Per-task success probabilities with the same mean and different spread."""
    out = {"constant": np.full(n_tasks, mean)}
    for name, conc in (("beta, wide", 2.0), ("beta, narrow", 20.0)):
        out[name] = rng.beta(mean * conc, (1 - mean) * conc, n_tasks)
    out["bimodal"] = (np.arange(n_tasks) < round(mean * n_tasks)).astype(float)   # each p_i is 0 or 1
    return out


# ---- cost -----------------------------------------------------------------------------------------------------------
def utility(quality: float, cost: float, lam: float) -> float:
    """U = Q - lambda * C."""
    return quality - lam * cost


# ---- horizon --------------------------------------------------------------------------------------------------------
def survival_exact(h: float, H, rho: float = 0.0):
    """S(H) = (1 - h(1-rho))^H."""
    return (1.0 - h * (1.0 - rho)) ** np.asarray(H)


def h50_exact(h: float, rho: float = 0.0) -> float:
    """The H at which S(H) = 1/2, from the exact (not small-h) hazard."""
    return log(2) / -log(1.0 - h * (1.0 - rho))


def first_failure_steps(h: float, n: int, rng, rho: float = 0.0, early_bad: float = 0.0, mult: float = 1.0):
    """Simulate n episodes step by step and return the step at which each hits an unrecoverable error.

    Each step errs with probability h. The agent catches and repairs a fraction rho of errors. With probability
    `early_bad` an episode starts with a bad early decision that multiplies the hazard by `mult` for every later step,
    so failures cluster instead of being independent.
    """
    hazard = np.full(n, h * (1.0 - rho))
    hazard[rng.random(n) < early_bad] *= mult
    return rng.geometric(np.clip(hazard, 1e-12, 1.0))


def empirical_survival(fail_steps, H) -> np.ndarray:
    """Fraction of episodes still alive after H steps (an H-step task succeeds iff the first failure is after H)."""
    fail_steps = np.sort(fail_steps)
    return 1.0 - np.searchsorted(fail_steps, np.asarray(H), side="right") / len(fail_steps)


def empirical_h50(fail_steps) -> float:
    return float(np.median(fail_steps))
