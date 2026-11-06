"""The Bradley-Terry model: from annotator noise to a sigmoid, and a reward fitted from comparisons.

An annotator sees two responses with true quality r_w and r_l, adds noise to each, and picks the larger:

    u(y) = r*(y) + eps_y,   prefer y_w  <=>  u(y_w) > u(y_l)

With Gumbel noise the choice probability is exactly sigmoid(r_w - r_l). With Gaussian noise it is the probit
Phi((r_w - r_l) / sqrt(2)) (Thurstone). Fitting scores by maximum likelihood is logistic regression on differences.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def sample_noise(kind: str, shape, scale: float, gen: torch.Generator) -> torch.Tensor:
    """Standard Gumbel or standard normal noise, times `scale`."""
    if kind == "gumbel":
        u = torch.rand(shape, generator=gen, dtype=torch.float64).clamp(1e-12, 1 - 1e-12)
        return -scale * torch.log(-torch.log(u))  # inverse CDF of Gumbel(0, 1): F(z) = exp(-exp(-z))
    if kind == "gaussian":
        return scale * torch.randn(shape, generator=gen, dtype=torch.float64)
    raise ValueError(kind)


def choice_rate(delta: float, kind: str, n: int, gen: torch.Generator, scale: float = 1.0) -> float:
    """Monte Carlo Pr[r_w + eps_w > r_l + eps_l] for a reward gap `delta` = r_w - r_l."""
    eps_w = sample_noise(kind, (n,), scale, gen)
    eps_l = sample_noise(kind, (n,), scale, gen)
    return (delta + eps_w > eps_l).double().mean().item()


def bt_prob(delta, scale: float = 1.0):
    """Bradley-Terry: sigmoid(delta / s). Noise scale s and reward scale are the same knob."""
    return torch.sigmoid(torch.as_tensor(delta, dtype=torch.float64) / scale)


def thurstone_prob(delta, scale: float = 1.0):
    """Gaussian noise on each response: Phi(delta / (s * sqrt(2)))."""
    z = torch.as_tensor(delta, dtype=torch.float64) / (scale * math.sqrt(2.0))
    return 0.5 * (1 + torch.erf(z / math.sqrt(2.0)))


def best_logistic_match(grid: torch.Tensor | None = None) -> tuple[float, float]:
    """The slope k for which sigmoid(k * delta) is closest to Thurstone's probit, and the largest gap left."""
    grid = torch.linspace(-8, 8, 4001, dtype=torch.float64) if grid is None else grid
    target = thurstone_prob(grid)
    best = (0.0, 1.0)
    for k in torch.linspace(0.5, 1.5, 2001, dtype=torch.float64):
        gap = (torch.sigmoid(k * grid) - target).abs().max().item()
        if gap < best[1]:
            best = (k.item(), gap)
    return best


def simulate_comparisons(r_true: torch.Tensor, n_pairs: int, gen: torch.Generator, scale: float = 1.0,
                         kind: str = "gumbel") -> tuple[torch.Tensor, torch.Tensor]:
    """Draw random pairs of distinct items and let a noisy annotator choose. Returns (winner, loser) indices."""
    n = r_true.numel()
    i = torch.randint(0, n, (n_pairs,), generator=gen)
    j = (i + torch.randint(1, n, (n_pairs,), generator=gen)) % n  # never equal to i
    u_i = r_true[i] + sample_noise(kind, (n_pairs,), scale, gen)
    u_j = r_true[j] + sample_noise(kind, (n_pairs,), scale, gen)
    i_wins = u_i > u_j
    return torch.where(i_wins, i, j), torch.where(i_wins, j, i)


def bt_loss(scores: torch.Tensor, winners: torch.Tensor, losers: torch.Tensor) -> torch.Tensor:
    """L_RM = -mean log sigmoid(r_w - r_l). Only differences of scores ever appear."""
    return -F.logsigmoid(scores[winners] - scores[losers]).mean()


def fit_bt(n_items: int, winners: torch.Tensor, losers: torch.Tensor, steps: int = 2000, lr: float = 0.05,
           l2: float = 0.0) -> torch.Tensor:
    """Maximum-likelihood scores, one free parameter per item, centred to mean zero at the end."""
    scores = torch.zeros(n_items, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([scores], lr=lr)
    for _ in range(steps):
        loss = bt_loss(scores, winners, losers) + l2 * scores.pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    s = scores.detach()
    return s - s.mean()


def bt_grad_magnitude(delta: torch.Tensor) -> torch.Tensor:
    """|d/d delta of -log sigmoid(delta)|, by autograd. The closed form is sigmoid(-delta)."""
    d = delta.detach().clone().double().requires_grad_(True)
    (-F.logsigmoid(d)).sum().backward()
    return d.grad.abs()
