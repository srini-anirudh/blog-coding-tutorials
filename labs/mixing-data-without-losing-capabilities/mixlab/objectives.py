"""Objectives over capability vectors, a fitted mixture surrogate, and a search over the simplex (post 7 to 9).

The surrogate is deliberately small. For capability j,

    S_j(w) = sum_i c_ij w_i + b_j * log(w_j + eps)

The linear part is the transfer-matrix idea (S = S0 + M^T w, with the intercept folded in because the weights
sum to one). The log term lets a capability saturate in its own data, which a purely linear model cannot do.
It is fitted by least squares on probe runs and then searched exhaustively on a grid, because with four sources
the grid is small.
"""

from __future__ import annotations

import itertools
import math
import random
from typing import Callable, Sequence

import torch

EPS = 0.02


def simplex_grid(k: int, step: float = 0.05) -> list[tuple[float, ...]]:
    n = round(1 / step)
    pts = []
    for c in itertools.product(range(n + 1), repeat=k - 1):
        if sum(c) <= n:
            pts.append(tuple(x / n for x in c) + ((n - sum(c)) / n,))
    return pts


def mean_obj(s: Sequence[float]) -> float:
    return sum(s) / len(s)


def maxmin_obj(s: Sequence[float]) -> float:
    return min(s)


def weighted_obj(u: Sequence[float]) -> Callable[[Sequence[float]], float]:
    return lambda s: sum(a * b for a, b in zip(u, s))


def constrained_obj(target: int, floors: Sequence[float]) -> Callable[[Sequence[float]], float]:
    """Maximise capability `target` subject to every other capability staying at or above its floor."""

    def f(s):
        if any(s[j] < floors[j] for j in range(len(s)) if j != target):
            return -math.inf
        return s[target]

    return f


def features(w: Sequence[float], j: int) -> list[float]:
    return list(w) + [math.log(w[j] + EPS)]


class Surrogate:
    """One small least-squares model per capability. Ridge keeps it stable with few probes."""

    def __init__(self, k: int, ridge: float = 1e-3, lo: float = -math.inf, hi: float = math.inf) -> None:
        self.k = k
        self.lo, self.hi = lo, hi  # accuracy cannot leave [0, 100], so the script clips predictions to that range
        self.ridge = ridge
        self.coef: list[torch.Tensor] = []

    def fit(self, ws: Sequence[Sequence[float]], ss: Sequence[Sequence[float]]) -> "Surrogate":
        self.coef = []
        for j in range(self.k):
            x = torch.tensor([features(w, j) for w in ws], dtype=torch.float64)
            y = torch.tensor([s[j] for s in ss], dtype=torch.float64)
            a = x.T @ x + self.ridge * torch.eye(x.shape[1], dtype=torch.float64)
            self.coef.append(torch.linalg.solve(a, x.T @ y))
        return self

    def predict(self, w: Sequence[float]) -> list[float]:
        raw = [float(torch.tensor(features(w, j), dtype=torch.float64) @ self.coef[j]) for j in range(self.k)]
        return [min(self.hi, max(self.lo, x)) for x in raw]

    def transfer_matrix(self) -> list[list[float]]:
        """The linear part c_ij as a sources x capabilities table (ignoring the own-data log term)."""
        return [[float(self.coef[j][i]) for j in range(self.k)] for i in range(self.k)]


def best_on_grid(predict: Callable, objective: Callable, grid) -> tuple[tuple[float, ...], list[float]]:
    best, best_val, best_s = None, -math.inf, None
    for w in grid:
        s = predict(w)
        v = objective(s)
        if v > best_val:
            best, best_val, best_s = w, v, s
    return best, best_s


def dirichlet_probes(k: int, n: int, seed: int, conc: float = 1.0) -> list[tuple[float, ...]]:
    """n random mixtures, uniform on the simplex when conc = 1."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        x = [rng.gammavariate(conc, 1.0) for _ in range(k)]
        out.append(tuple(round(v / sum(x), 3) for v in x))
    return out


def linear_scalarisation_hits(points: Sequence[Sequence[float]], n_dirs: int = 181) -> set[int]:
    """Which of a set of 2-D capability vectors can ever win a weighted sum u1*s1 + u2*s2, for any u >= 0.

    Points in a dent of a non-convex Pareto front never win: the maximiser of a linear function lies on the
    convex hull. Max-min can pick them.
    """
    hits = set()
    for t in range(n_dirs):
        a = t / (n_dirs - 1)
        u = (a, 1 - a)
        hits.add(max(range(len(points)), key=lambda i: u[0] * points[i][0] + u[1] * points[i][1]))
    return hits
