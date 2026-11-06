"""Amdahl's law with a coordination term (the post's Amdahl, Brooks, Coase section).

    T(N) = T1 [(1 - f) + f / N] + kappa * N^alpha

The first term is Amdahl. The second is the cost of briefing agents, reading their output
and resolving their conflicts. Together they have an interior minimum N*.
"""

from __future__ import annotations

import math


def amdahl_speedup(n: float, f: float) -> float:
    return 1.0 / ((1 - f) + f / n)


def amdahl_limit(f: float) -> float:
    return 1.0 / (1 - f)


def wall_time(n: float, f: float, kappa: float, alpha: float, t1: float = 1.0) -> float:
    return t1 * ((1 - f) + f / n) + kappa * n**alpha


def real_speedup(n: float, f: float, kappa: float, alpha: float) -> float:
    return 1.0 / wall_time(n, f, kappa, alpha)


def best_n(f: float, kappa: float, alpha: float, n_max: int = 256) -> int:
    """Integer N that minimises T(N), searched over 1..n_max."""
    return min(range(1, n_max + 1), key=lambda n: wall_time(n, f, kappa, alpha))


def best_n_continuous(f: float, kappa: float, alpha: float) -> float:
    """Stationary point of T(N): -f/N^2 + kappa*alpha*N^(alpha-1) = 0  =>  N* = (f/(kappa*alpha))^(1/(alpha+1))."""
    if kappa <= 0:
        return math.inf
    return (f / (kappa * alpha)) ** (1.0 / (alpha + 1))


def break_even_n(f: float, kappa: float, alpha: float, n_max: int = 10_000) -> int | None:
    """Smallest N > 1 past which the system is slower than one agent (T(N) > T(1)), if any."""
    t1 = wall_time(1, f, kappa, alpha)
    for n in range(2, n_max + 1):
        if wall_time(n, f, kappa, alpha) > t1:
            return n
    return None
