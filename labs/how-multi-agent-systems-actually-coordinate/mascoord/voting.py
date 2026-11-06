"""Voting, best-of-N and the correlation ceiling.

The post's model: each agent is right with probability p. With probability rho the whole
ensemble shares one blind spot and gives one shared answer (right with probability p).
Otherwise the agents are independent. Majority-vote accuracy is

    A(N) = rho * p + (1 - rho) * P[Bin(N, p) > N/2]        ->   1 - rho * (1 - p)

The ceiling depends on rho, not on N. This file gives the closed form, a Monte Carlo check
that simulates the agents one by one, and the best-of-N alternative with a perfect verifier.
"""

from __future__ import annotations

import math

import numpy as np


def p_majority(n: int, p: float) -> float:
    """P[Bin(n, p) > n/2]: a strict majority of n independent agents is right.

    A tie (even n) counts as wrong, the same rule the post's figure uses.
    """
    need = n // 2 + 1
    return sum(math.comb(n, k) * p**k * (1 - p) ** (n - k) for k in range(need, n + 1))


def vote_accuracy(n: int, p: float, rho: float) -> float:
    """Closed-form A(N) under the common-mode model."""
    return rho * p + (1 - rho) * p_majority(n, p)


def vote_ceiling(p: float, rho: float) -> float:
    """lim A(N) as N -> infinity, for p > 1/2."""
    return 1 - rho * (1 - p)


def p_any(n: int, p: float) -> float:
    """1 - (1 - p)^n: at least one of n independent tries succeeds."""
    return 1 - (1 - p) ** n


def best_of_n_accuracy(n: int, p: float, rho: float) -> float:
    """Best-of-N with a perfect verifier under the same common-mode model.

    A verifier helps whenever any sample is right. In a common-mode episode every sample is
    the same answer, so the verifier has nothing better to pick.
    """
    return rho * p + (1 - rho) * p_any(n, p)


def simulate(n: int, p: float, rho: float, trials: int = 200_000, n_wrong_answers: int = 50,
             seed: int = 0) -> dict:
    """Monte Carlo of the same model, one agent at a time.

    Wrong agents pick one of `n_wrong_answers` wrong answers at random, so wrong votes can
    split. That is slightly kinder to voting than "all wrong answers agree", and it lets a
    plurality vote be measured as well as a strict majority.
    Returns strict-majority, plurality and best-of-N accuracy.
    """
    rng = np.random.default_rng(seed)
    common = rng.random(trials) < rho                       # (trials,) shared blind spot?
    shared_right = rng.random(trials) < p                   # (trials,) the shared answer is right
    indep_right = rng.random((trials, n)) < p               # (trials, n)
    right = np.where(common[:, None], shared_right[:, None], indep_right)   # (trials, n)
    wrong_ids = rng.integers(1, n_wrong_answers + 1, size=(trials, n))
    shared_wrong = rng.integers(1, n_wrong_answers + 1, size=trials)
    wrong_ids = np.where(common[:, None], shared_wrong[:, None], wrong_ids)
    answers = np.where(right, 0, wrong_ids)                 # answer 0 is the correct one

    n_right = right.sum(1)
    strict = n_right > n / 2
    # plurality: answer 0 wins if no wrong answer has as many votes (ties count as wrong)
    counts = np.zeros((trials, n_wrong_answers + 1), dtype=np.int32)
    np.add.at(counts, (np.arange(trials)[:, None], answers), 1)
    best_wrong = counts[:, 1:].max(1)
    plurality = counts[:, 0] > best_wrong
    any_right = n_right > 0
    return {"strict_majority": float(strict.mean()), "plurality": float(plurality.mean()),
            "best_of_n": float(any_right.mean())}


def implied_rho(ceiling: float, p: float) -> float:
    """Invert A_inf = 1 - rho (1 - p) to read rho off a measured plateau."""
    return (1 - ceiling) / (1 - p)


def mixed_team_accuracy(n_strong: int, n_weak: int, p_strong: float, p_weak: float,
                        rho_same: float, rho_mixed: float) -> float:
    """Self-MoA versus MoA, inside the same toy model.

    A team of one model family shares blind spots at rate rho_same. Mixing in a second
    family lowers the shared-blind-spot rate to rho_mixed but brings weaker members. In a
    common-mode episode the team gives the answer of a strong member (p_strong). Otherwise
    members vote independently, each with its own accuracy. Computed exactly by convolving
    the two binomials.
    """
    n = n_strong + n_weak
    rho = rho_same if n_weak == 0 else rho_mixed
    dist = np.zeros(n + 1)
    for a in range(n_strong + 1):
        pa = math.comb(n_strong, a) * p_strong**a * (1 - p_strong) ** (n_strong - a)
        for b in range(n_weak + 1):
            pb = math.comb(n_weak, b) * p_weak**b * (1 - p_weak) ** (n_weak - b)
            dist[a + b] += pa * pb
    indep = dist[n // 2 + 1:].sum()
    return rho * p_strong + (1 - rho) * float(indep)
