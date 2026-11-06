"""How per-step accuracy compounds over a horizon, with and without recovery.

Without recovery a task of K steps succeeds with probability p**K. With recovery the process is an
absorbing Markov chain over three states: on track (T), recoverable error (E), dead (D).
    T -> T with p,  T -> E with 1 - p
    E -> T with q,  E -> D with 1 - q
    D -> D
Success means being on track after K steps. With q = 0 this is exactly p**K.
"""

from __future__ import annotations

import random

import torch


def required_p(target: float, k: int) -> float:
    """The per-step accuracy needed to keep task success at `target` over k steps: target ** (1 / k)."""
    return target ** (1.0 / k)


def transition(p: float, q: float) -> torch.Tensor:
    # Rows are the current state, columns the next state, in the order T, E, D.
    return torch.tensor([[p, 1 - p, 0.0], [q, 0.0, 1 - q], [0.0, 0.0, 1.0]], dtype=torch.float64)


def success_exact(p: float, q: float, k: int) -> float:
    start = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    return float((start @ torch.linalg.matrix_power(transition(p, q), k))[0])


def success_monte_carlo(p: float, q: float, k: int, trials: int, seed: int = 0) -> float:
    rng = random.Random(seed)
    ok = 0
    for _ in range(trials):
        state = "T"
        for _ in range(k):
            u = rng.random()
            if state == "T":
                state = "T" if u < p else "E"
            elif state == "E":
                state = "T" if u < q else "D"
            else:
                break
        ok += state == "T"
    return ok / trials
