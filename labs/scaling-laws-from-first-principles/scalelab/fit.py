"""Fitting scaling laws: a power law with a floor, the IsoFLOP procedure, and the parametric (N, D) fit.

No SciPy: the parametric fit runs every initialisation at once as one batched PyTorch problem.
"""

from __future__ import annotations

import itertools

import numpy as np
import torch

from scalelab.allocation import LossLaw


def power_law(x, y) -> tuple[float, float]:
    """Fit y = A x^-alpha by a straight line in log-log space. Returns (A, alpha)."""
    slope, icept = np.polyfit(np.log(x), np.log(y), 1)
    return float(np.exp(icept)), float(-slope)


def power_law_with_floor(x, y, floors: int = 400) -> tuple[float, float, float]:
    """Fit y = E + A x^-alpha. Try many floors E below min(y); for each, the rest is a log-log line.

    Keeps the E whose fit has the smallest squared error in log y. Returns (E, A, alpha).
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    best = None
    for E in np.linspace(0.0, y.min() * 0.999, floors):
        A, alpha = power_law(x, y - E)
        err = np.sum((np.log(E + A * x ** -alpha) - np.log(y)) ** 2)
        if best is None or err < best[0]:
            best = (err, E, A, alpha)
    return best[1], best[2], best[3]


def isoflop_minimum(n, loss) -> tuple[float, float]:
    """Approach 2: fit a parabola to loss against log10(N) at one budget and return its vertex (N_opt, L_min)."""
    x = np.log10(np.asarray(n, float))
    c2, c1, c0 = np.polyfit(x, np.asarray(loss, float), 2)
    if c2 <= 0:
        raise ValueError("valley is not convex: the sweep does not bracket the minimum")
    xv = -c1 / (2 * c2)
    return float(10 ** xv), float(c0 - c1 * c1 / (4 * c2))


def huber(r: torch.Tensor, delta: float) -> torch.Tensor:
    a = r.abs()
    return torch.where(a <= delta, 0.5 * r * r, delta * (a - 0.5 * delta))


def fit_parametric(n, d, loss, delta: float = 1e-3, steps: int = 3000) -> tuple[LossLaw, float]:
    """Approach 3 (Hoffmann et al.): fit log L = LSE(a - alpha log N, b - beta log D, e) under a Huber loss.

    A = e^a, B = e^b, E = e^e. Like the paper, start from a grid of initialisations and keep the best end point.
    Every start is optimised at once (Adam, then L-BFGS on the best). Returns (law, final objective).
    """
    ln = torch.tensor(np.log(n), dtype=torch.float64)
    ld = torch.tensor(np.log(d), dtype=torch.float64)
    ll = torch.tensor(np.log(loss), dtype=torch.float64)
    grid = list(itertools.product([0, 5, 10, 15, 20], [0, 5, 10, 15, 20], [-1, -0.5, 0, 0.5, 1],
                                  [0.1, 0.4, 0.8, 1.5], [0.1, 0.4, 0.8, 1.5]))
    p = torch.tensor(grid, dtype=torch.float64, requires_grad=True)  # (starts, 5): a, b, e, alpha, beta

    def objective(q: torch.Tensor) -> torch.Tensor:  # (starts, 5) -> (starts,)
        a, b, e, al, be = (q[:, i : i + 1] for i in range(5))
        pred = torch.logsumexp(torch.stack([a - al * ln, b - be * ld, e.expand(-1, ln.numel())]), dim=0)
        return huber(pred - ll, delta).sum(dim=1)

    opt = torch.optim.Adam([p], lr=0.05)
    for _ in range(steps):
        opt.zero_grad()
        objective(p).sum().backward()
        opt.step()
        with torch.no_grad():
            p[:, 3:].clamp_(min=1e-3)

    best = p.detach()[int(torch.argmin(objective(p.detach())))].clone().unsqueeze(0).requires_grad_(True)
    lbfgs = torch.optim.LBFGS([best], lr=1.0, max_iter=500, tolerance_grad=1e-12, tolerance_change=1e-15,
                              line_search_fn="strong_wolfe")

    def closure():
        lbfgs.zero_grad()
        f = objective(best).sum()
        f.backward()
        return f

    lbfgs.step(closure)
    a, b, e, al, be = best.detach()[0].tolist()
    final = float(objective(best.detach()).sum())
    return LossLaw(E=float(np.exp(e)), A=float(np.exp(a)), B=float(np.exp(b)), alpha=al, beta=be), final
