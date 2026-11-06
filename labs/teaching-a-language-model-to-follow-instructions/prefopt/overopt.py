"""Goodhart's law in miniature: a gold reward, a proxy fitted to its comparisons, and two ways to optimise the proxy.

This copies the shape of Gao, Schulman and Hilton (2022) at toy scale. A completion is a vector x ~ N(0, I) under
the reference policy. A fixed "gold" function stands in for the human. A small "proxy" network is fitted to
Bradley-Terry labels that the gold function produced on reference samples. Then:

  best-of-n:  draw n samples from the reference, keep the one the proxy likes best.
              KL(bon || ref) = log n - (n - 1) / n  (exact for continuous samples).
  KL tilt:    pi_beta(x) ∝ ref(x) exp(proxy(x) / beta), the KL-regularised optimum, computed by reweighting a large
              pool of reference samples. This stands in for RL, which aims at the same distribution.

Gold reward is tracked against d = sqrt(KL), and the paper's functional forms are fitted to the measured curves.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

DIM = 4
A = torch.tensor([2.0, 1.0, 0.5, 0.0], dtype=torch.float64)  # gold rises along these directions ...
B = torch.tensor([0.5, 0.5, 0.25, 0.25], dtype=torch.float64)  # ... and is pulled back by these curvatures


def gold(x: torch.Tensor) -> torch.Tensor:
    """r*(x) = sum_k a_k x_k - b_k x_k^2. It peaks at a finite distance from the reference's mode."""
    return x @ A - (x**2) @ B


class Proxy(nn.Module):
    """A one-hidden-layer reward model. width=0 means a linear model."""

    def __init__(self, width: int, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.net = nn.Linear(DIM, 1) if width == 0 else nn.Sequential(nn.Linear(DIM, width), nn.Tanh(), nn.Linear(width, 1))
        self.double()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def make_labels(n_pairs: int, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Pairs of reference samples, labelled by a Bradley-Terry annotator on the gold reward. Returns (x_w, x_l)."""
    a = torch.randn(n_pairs, DIM, generator=gen, dtype=torch.float64)
    b = torch.randn(n_pairs, DIM, generator=gen, dtype=torch.float64)
    a_wins = torch.rand(n_pairs, generator=gen, dtype=torch.float64) < torch.sigmoid(gold(a) - gold(b))
    m = a_wins.unsqueeze(-1)
    return torch.where(m, a, b), torch.where(m, b, a)


def fit_proxy(n_pairs: int, width: int, seed: int = 0, epochs: int = 300, lr: float = 0.01) -> Proxy:
    """Fit a proxy with the Bradley-Terry loss on n_pairs gold-labelled comparisons (full batch, Adam)."""
    gen = torch.Generator().manual_seed(seed)
    x_w, x_l = make_labels(n_pairs, gen)
    proxy = Proxy(width, seed)
    opt = torch.optim.Adam(proxy.parameters(), lr=lr)
    for _ in range(epochs):
        loss = -F.logsigmoid(proxy(x_w) - proxy(x_l)).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return proxy.eval()


def bon_kl(n: int) -> float:
    return math.log(n) - (n - 1) / n


@torch.no_grad()
def best_of_n_curve(proxy: Proxy, ns: list[int], trials: int, seed: int = 1) -> list[dict]:
    """Mean gold and proxy reward of the proxy's favourite among n reference samples."""
    gen = torch.Generator().manual_seed(seed)
    rows = []
    for n in ns:
        picks = []
        chunk = max(1, 2**18 // n)  # keep each block near 262k samples so the proxy's activations stay small
        for start in range(0, trials, chunk):
            t = min(chunk, trials - start)
            x = torch.randn(t, n, DIM, generator=gen, dtype=torch.float64)
            picks.append(x[torch.arange(t), proxy(x).argmax(-1)])
        pick = torch.cat(picks)
        rows.append({"n": n, "d": math.sqrt(bon_kl(n)), "gold": gold(pick).mean().item(),
                     "proxy": proxy(pick).mean().item()})
    return rows


@torch.no_grad()
def tilt_curve(proxy: Proxy, betas: list[float], pool: int = 400_000, seed: int = 2) -> list[dict]:
    """Self-normalised importance weights w ∝ exp(proxy / beta) over a reference pool.

    KL(pi_beta || ref) on the pool is sum w log(N w). The effective sample size says when the pool runs out.
    """
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(pool, DIM, generator=gen, dtype=torch.float64)
    s, g = proxy(x), gold(x)
    rows = []
    for beta in betas:
        logw = torch.log_softmax(s / beta, 0)
        w = logw.exp()
        k = (w * (logw + math.log(pool))).sum().item()
        rows.append({"beta": beta, "d": math.sqrt(max(k, 0.0)), "gold": (w * g).sum().item(),
                     "proxy": (w * s).sum().item(), "ess": (1.0 / (w**2).sum()).item()})
    return rows


def fit_form(d: torch.Tensor, r: torch.Tensor, form: str) -> tuple[float, float, float]:
    """Least squares for Gao et al.'s forms. R/d = alpha - beta*d (best-of-n) or alpha - beta*log d (RL).

    Both are linear in (alpha, beta) after dividing by d. Returns alpha, beta and R^2 on R itself.
    """
    d, r = d.double(), r.double()
    feat = d if form == "bo" else torch.log(d)
    X = torch.stack([d, -d * feat], -1)  # R = alpha * d - beta * d * feat
    coef = torch.linalg.lstsq(X, r.unsqueeze(-1)).solution.squeeze(-1)
    pred = X @ coef
    r2 = 1 - ((r - pred) ** 2).sum() / ((r - r.mean()) ** 2).sum()
    return coef[0].item(), coef[1].item(), r2.item()
