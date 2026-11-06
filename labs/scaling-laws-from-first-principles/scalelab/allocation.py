"""The allocation problem: minimise L(N, D) = E + A N^-a + B D^-b subject to a compute budget.

Everything here is closed-form algebra from the post (sections 5, 8, 9 and 14), plus brute-force checks of it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LossLaw:
    E: float
    A: float
    B: float
    alpha: float
    beta: float

    def __call__(self, N, D):
        return self.E + self.A * np.power(N, -self.alpha) + self.B * np.power(D, -self.beta)

    @property
    def a(self) -> float:
        """N* grows as C^a."""
        return self.beta / (self.alpha + self.beta)

    @property
    def b(self) -> float:
        """D* grows as C^b. a + b = 1 because C = 6ND."""
        return self.alpha / (self.alpha + self.beta)


# Hoffmann et al. (2022), Approach 3, as published. The post uses these in every figure.
HOFFMANN = LossLaw(E=1.69, A=406.4, B=410.7, alpha=0.34, beta=0.28)


def optimal(law: LossLaw, C: float, k: float = 6.0) -> tuple[float, float]:
    """Compute-optimal (N*, D*) for budget C = k N D.

    Setting dL/dN = 0 with D = C/(kN) gives N^(a+b) = (alpha A / (beta B)) (C/k)^beta.
    """
    g = (law.alpha * law.A / (law.beta * law.B)) ** (1.0 / (law.alpha + law.beta))
    n = g * (C / k) ** law.a
    return n, C / (k * n)


def optimal_by_search(law: LossLaw, C: float, k: float = 6.0, points: int = 200_001) -> tuple[float, float]:
    """The same optimum found by evaluating the loss along the whole budget curve, no calculus."""
    n = np.logspace(3, 15, points)
    i = int(np.argmin(law(n, C / (k * n))))
    return float(n[i]), float(C / (k * n[i]))


def shortfalls(law: LossLaw, N: float, D: float) -> tuple[float, float]:
    """The two reducible terms T_N = A N^-alpha and T_D = B D^-beta."""
    return law.A * N ** -law.alpha, law.B * D ** -law.beta


def lifetime_optimal(law: LossLaw, target: float, inference_tokens: float, points: int = 20_001):
    """Cheapest (N, D) that reaches `target` loss when lifetime cost is 6 N D + 2 N T_inf (section 8).

    For each N on a grid, D is whatever reaches the target exactly; then pick the N with the smallest cost.
    Returns (N, D, cost) or None if the target is below the reachable floor.
    """
    n = np.logspace(5, 14, points)
    room = target - law.E - law.A * n ** -law.alpha  # what the data term may still contribute
    ok = room > 0
    if not ok.any():
        return None
    n = n[ok]
    d = (law.B / room[ok]) ** (1.0 / law.beta)
    cost = 6 * n * d + 2 * n * inference_tokens
    i = int(np.argmin(cost))
    return float(n[i]), float(d[i]), float(cost[i])


def effective_tokens(unique: float, epochs: float, r_star: float = 15.4) -> float:
    """Muennighoff et al. (2023): D' = U + U R* (1 - exp(-R/R*)), with R = epochs - 1 repeats past the first pass."""
    r = max(epochs - 1.0, 0.0)
    return unique + unique * r_star * (1.0 - math.exp(-r / r_star))


@dataclass
class Plan:
    budget: float
    n: float
    d: float
    epochs: float
    effective_d: float
    lifetime_n: float | None


def plan(gpus: int, hours: float, peak_flops: float, mfu: float, unique_tokens: float,
         inference_tokens: float = 0.0, law: LossLaw = HOFFMANN) -> Plan:
    """The post's Figure 16 as a function: cluster -> budget -> allocation -> corpus check -> serving check."""
    C = gpus * hours * 3600 * peak_flops * mfu
    n, d = optimal(law, C)
    epochs = d / unique_tokens
    life = None
    if inference_tokens > 0:
        res = lifetime_optimal(law, float(law(n, d)), inference_tokens)
        life = res[0] if res else None
    return Plan(C, n, d, epochs, effective_tokens(unique_tokens, epochs) if epochs > 1 else d, life)
