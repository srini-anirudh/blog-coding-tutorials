"""Section 3 of the post: outcome rewards reinforce lucky trajectories, and Monte Carlo step labels measure the
completion model rather than the step.

A derivation has `steps` steps. At each step the policy makes its first mistake with probability `p_err`. After a
mistake, the final answer can still come out right (an error that cancels) with probability `p_lucky`.

Monte Carlo step labelling: from the prefix that ends at step t, a completion model finishes the solution `m`
times. From a clean prefix it reaches the right answer with probability `a_clean`. From a broken prefix it still
reaches the right answer with probability `a_rescue`. The "hard" label calls a step good when any completion is
right; the "soft" label calls it good when more than half are.
"""

from __future__ import annotations

import numpy as np


def sample_derivations(n: int, steps: int, p_err: float, p_lucky: float, seed: int = 0):
    """Returns (step_ok (n, steps) bool, answer_ok (n,) bool). step_ok is False from the first mistake on."""
    rng = np.random.default_rng(seed)
    first_err = rng.geometric(p_err, size=n) - 1  # index of the first wrong step; >= steps means none
    step_ok = np.arange(steps)[None, :] < first_err[:, None]
    clean = first_err >= steps
    answer_ok = clean | (rng.random(n) < p_lucky)
    return step_ok, answer_ok


def outcome_false_positive_rate(step_ok: np.ndarray, answer_ok: np.ndarray) -> float:
    """Among trajectories with reward 1, the fraction that contain a wrong step."""
    rewarded = answer_ok
    return float((~step_ok.all(axis=1))[rewarded].mean())


def mc_step_labels(step_ok: np.ndarray, a_clean: float, a_rescue: float, m: int, seed: int = 0):
    """Monte Carlo labels for every step. Returns (hard, soft), both bool arrays shaped like step_ok."""
    rng = np.random.default_rng(seed)
    prob = np.where(step_ok, a_clean, a_rescue)
    hits = rng.binomial(m, prob)
    return hits > 0, hits > m / 2


def label_accuracy(step_ok: np.ndarray, label: np.ndarray) -> dict[str, float]:
    """How often good steps are labelled good, and how often bad steps are labelled bad."""
    return {
        "good_steps_called_good": float(label[step_ok].mean()),
        "bad_steps_called_bad": float((~label[~step_ok]).mean()),
    }


def hard_label_good_prob(a: float, m: int) -> float:
    """Closed form: P(at least one of m completions is right) = 1 - (1 - a)^m."""
    return 1.0 - (1.0 - a) ** m
