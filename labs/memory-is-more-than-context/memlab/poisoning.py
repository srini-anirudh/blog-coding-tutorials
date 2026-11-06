"""Section 16, "When memory poisons itself": the experience-following fixed point.

The post's toy model: with probability phi the agent copies a retrieved record, otherwise it
reasons fresh with error eps0. A write gate keeps a correct episode with probability beta and
a wrong one with probability alpha, and rho = alpha / beta. The closed form below is the post's.
`simulate` runs the loop with real random draws and a real store, so the two can be compared.
"""

from __future__ import annotations

import math
import random


def e_star(rho: float, phi: float, eps0: float) -> float:
    """The post's root of e = phi * w(e) + (1 - phi) * eps0, with w(e) = rho e / (rho e + 1 - e)."""
    k, c = rho - 1.0, (1.0 - phi) * eps0
    if abs(k) < 1e-12:
        return eps0
    b = 1.0 - phi * rho - c * k
    return (-b + math.sqrt(b * b + 4.0 * k * c)) / (2.0 * k)


def wrong_fraction(e: float, rho: float) -> float:
    return rho * e / (rho * e + 1.0 - e)


def iterate(rho: float, phi: float, eps0: float, e0: float | None = None, steps: int = 2000) -> float:
    """Solve the same fixed point by iterating the map e -> phi w(e) + (1 - phi) eps0."""
    e = eps0 if e0 is None else e0
    for _ in range(steps):
        e = phi * wrong_fraction(e, rho) + (1.0 - phi) * eps0
    return e


def simulate(rho: float, phi: float, eps0: float, tasks: int = 40000, beta: float = 0.5,
             seed: int = 0, tail: float = 0.5) -> dict:
    """Run the agent loop. The store starts empty and only holds what the gate let in.

    Each task: if the store is not empty, copy a random stored record with probability phi
    (its outcome is the record's), else reason fresh (wrong with probability eps0). Then the
    gate stores the episode with probability beta if correct, alpha = rho * beta if wrong.
    Returns the error rate over the last `tail` share of tasks, and the store's wrong fraction.
    """
    alpha = rho * beta
    if alpha > 1:
        beta, alpha = 1.0 / rho, 1.0
    rng = random.Random(seed)
    store: list[bool] = []  # True means the stored episode was wrong
    wrong_in_store = 0
    errors_tail = 0
    start_tail = int(tasks * (1 - tail))
    for i in range(tasks):
        if store and rng.random() < phi:
            wrong = store[rng.randrange(len(store))]
        else:
            wrong = rng.random() < eps0
        if rng.random() < (alpha if wrong else beta):
            store.append(wrong)
            wrong_in_store += wrong
        if i >= start_tail:
            errors_tail += wrong
    return {"error": errors_tail / (tasks - start_tail), "store_wrong": wrong_in_store / max(len(store), 1),
            "store_size": len(store)}
