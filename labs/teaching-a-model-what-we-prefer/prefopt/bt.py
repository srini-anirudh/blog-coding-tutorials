"""Bradley-Terry: from annotator noise to a sigmoid, and the reward-model loss it gives.

An annotator sees two responses with true quality r_w and r_l, adds independent noise to each, and picks the larger.
With Gumbel(0, s) noise, P(pick w) = sigmoid((r_w - r_l) / s). With Gaussian noise it is the probit Phi(delta / sqrt 2).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def simulate_choices(delta: torch.Tensor, n: int, noise: str = "gumbel", scale: float = 1.0, seed: int = 0):
    """Fraction of n simulated annotators who prefer w, for each reward gap in delta (shape (G,))."""
    g = torch.Generator().manual_seed(seed)
    shape = (delta.numel(), n)
    if noise == "gumbel":
        u = torch.rand(2, *shape, generator=g, dtype=torch.float64).clamp_min(1e-300)
        eps = -torch.log(-torch.log(u))  # inverse CDF of Gumbel(0, 1)
    elif noise == "gaussian":
        eps = torch.randn(2, *shape, generator=g, dtype=torch.float64)
    else:
        raise ValueError(noise)
    u_w = delta.double()[:, None] + scale * eps[0]
    u_l = scale * eps[1]
    return (u_w > u_l).double().mean(-1)


def probit(delta: torch.Tensor) -> torch.Tensor:
    """Thurstone: the difference of two N(0, 1) has variance 2."""
    return 0.5 * (1 + torch.erf(delta.double() / 2))  # Phi(d / sqrt 2) = (1 + erf(d / 2)) / 2


def bt_loss(r_w: torch.Tensor, r_l: torch.Tensor) -> torch.Tensor:
    """-log sigmoid(r_w - r_l), averaged. logsigmoid is the stable form of log(1 / (1 + e^-x))."""
    return -F.logsigmoid(r_w - r_l).mean()


def sample_comparisons(r_true: torch.Tensor, n_pairs: int, scale: float = 1.0, seed: int = 0):
    """Random pairs of items labelled by Bradley-Terry annotators. Returns (winner, loser) index tensors."""
    g = torch.Generator().manual_seed(seed)
    k = r_true.numel()
    a = torch.randint(0, k, (n_pairs,), generator=g)
    b = (a + torch.randint(1, k, (n_pairs,), generator=g)) % k  # never the same item twice
    p_a = torch.sigmoid((r_true[a] - r_true[b]) / scale)
    a_wins = torch.rand(n_pairs, generator=g) < p_a
    return torch.where(a_wins, a, b), torch.where(a_wins, b, a)


def fit_scores(winners: torch.Tensor, losers: torch.Tensor, k: int, steps: int = 2000, lr: float = 0.1,
               init: torch.Tensor | None = None) -> torch.Tensor:
    """Maximum-likelihood Bradley-Terry scores for k items, one free scalar each, by full-batch Adam."""
    r = (torch.zeros(k) if init is None else init.clone()).requires_grad_(True)
    opt = torch.optim.Adam([r], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        bt_loss(r[winners], r[losers]).backward()
        opt.step()
    return r.detach()


def logistic_probit_gap(c: float, grid: torch.Tensor) -> float:
    """Largest |sigmoid(c * d) - Phi(d / sqrt 2)| over the grid: how different the two links are after rescaling."""
    return (torch.sigmoid(c * grid.double()) - probit(grid)).abs().max().item()


def best_logistic_rescale(grid: torch.Tensor) -> tuple[float, float]:
    """The scale c that makes the logistic link closest to the probit link (golden-section search)."""
    lo, hi = 0.5, 2.0
    phi = (math.sqrt(5) - 1) / 2
    for _ in range(80):
        a, b = hi - phi * (hi - lo), lo + phi * (hi - lo)
        if logistic_probit_gap(a, grid) < logistic_probit_gap(b, grid):
            hi = b
        else:
            lo = a
    c = (lo + hi) / 2
    return c, logistic_probit_gap(c, grid)
