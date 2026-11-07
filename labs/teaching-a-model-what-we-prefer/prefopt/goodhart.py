"""A small version of the Gao, Schulman and Hilton setup: a gold reward labels pairs, a proxy is fitted to the
labels, and a policy is optimised against the proxy while the gold reward is watched.

The "completions" are a fixed pool of feature vectors. The reference policy puts most of its mass near the centre
of the pool, so the proxy only ever sees labels there. Coordinate 0 plays the part of response length: gold likes a
little more of it and then less, which the proxy cannot learn from data that rarely goes past the peak.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .bt import bt_loss
from .tilt import best_of_n, kl, tilt


@dataclass
class World:
    feats: torch.Tensor  # (N, d) the pool of completions
    ref: torch.Tensor  # (N,) reference policy over the pool
    gold: torch.Tensor  # (N,) the reward we actually care about


def make_world(n: int = 4000, d: int = 6, pool_std: float = 2.0, seed: int = 0) -> World:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=g, dtype=torch.float64) * pool_std
    # ref ~ N(0, I) restricted to the pool: weight each pool point by N(0, I) / N(0, pool_std^2 I).
    logw = -0.5 * (x**2).sum(-1) * (1 - 1 / pool_std**2)
    ref = torch.softmax(logw, 0)
    # Gold: a bounded random function of the content coordinates, plus a length term that peaks at length = 1.
    w1 = torch.randn(d - 1, 32, generator=g, dtype=torch.float64)
    w2 = torch.randn(32, generator=g, dtype=torch.float64) / 32**0.5
    content = torch.tanh(x[:, 1:] @ w1 / (d - 1) ** 0.5) @ w2
    length = x[:, 0]
    gold = content + 0.8 * length - 0.4 * length**2
    return World(x, ref, gold)


def sample_pairs(world: World, m: int, seed: int = 0):
    """m comparisons between two independent draws from ref, labelled by Bradley-Terry on the gold reward."""
    g = torch.Generator().manual_seed(seed)
    a = torch.multinomial(world.ref, m, replacement=True, generator=g)
    b = torch.multinomial(world.ref, m, replacement=True, generator=g)
    a_wins = torch.rand(m, generator=g, dtype=torch.float64) < torch.sigmoid(world.gold[a] - world.gold[b])
    return torch.where(a_wins, a, b), torch.where(a_wins, b, a)


class Proxy(nn.Module):
    """A ReLU MLP reward model. ReLU nets extrapolate linearly, which is the failure we want to see."""

    def __init__(self, d: int, width: int, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.net = nn.Sequential(nn.Linear(d, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU(),
                                 nn.Linear(width, 1)).double()

    def forward(self, x):
        return self.net(x).squeeze(-1)


def fit_proxy(world: World, winners, losers, width: int, steps: int = 1500, lr: float = 3e-3, seed: int = 0):
    proxy = Proxy(world.feats.shape[1], width, seed)
    opt = torch.optim.AdamW(proxy.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(steps):
        r = proxy(world.feats)
        loss = bt_loss(r[winners], r[losers])
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return proxy(world.feats)


def pairwise_accuracy(world: World, score: torch.Tensor, m: int = 20_000, seed: int = 99) -> float:
    """How often the score orders two fresh ref draws the way gold does."""
    g = torch.Generator().manual_seed(seed)
    a = torch.multinomial(world.ref, m, replacement=True, generator=g)
    b = torch.multinomial(world.ref, m, replacement=True, generator=g)
    keep = world.gold[a] != world.gold[b]
    return ((score[a] > score[b]) == (world.gold[a] > world.gold[b]))[keep].double().mean().item()


def optimisation_curves(world: World, proxy: torch.Tensor):
    """Gold and proxy reward against d = sqrt(KL) for best-of-n and for the exact KL-regularised optimum."""
    base_gold, base_len = (world.ref * world.gold).sum(), (world.ref * world.feats[:, 0]).sum()
    out = {"bon": [], "kl": []}
    for n in [2**k for k in range(0, 21)]:
        p = best_of_n(world.ref, proxy, n)
        out["bon"].append(_point(p, world, proxy, base_gold, base_len, n=n))
    for beta in torch.logspace(2.0, -0.5, 26).tolist():
        p = tilt(world.ref, proxy, beta)
        out["kl"].append(_point(p, world, proxy, base_gold, base_len, beta=beta))
    return out


def _point(p, world, proxy, base_gold, base_len, **kw):
    return {**kw, "d": kl(p, world.ref).clamp_min(0).sqrt().item(),
            "gold": ((p * world.gold).sum() - base_gold).item(),
            "proxy": ((p * proxy).sum() - (world.ref * proxy).sum()).item(),
            "length": ((p * world.feats[:, 0]).sum() - base_len).item()}


def fit_form(points: list[dict], form: str):
    """Least squares for Gao et al.'s forms: bon  R = a d - b d^2;  rl  R = a d - b d log d. Returns a, b, R^2."""
    pts = [q for q in points if q["d"] > 1e-6]
    d = torch.tensor([q["d"] for q in pts], dtype=torch.float64)
    y = torch.tensor([q["gold"] for q in pts], dtype=torch.float64)
    second = d**2 if form == "bon" else d * d.log()
    a_mat = torch.stack([d, -second], 1)
    coef = torch.linalg.lstsq(a_mat, y[:, None]).solution.squeeze(1)
    resid = y - a_mat @ coef
    r2 = 1 - (resid**2).sum() / ((y - y.mean()) ** 2).sum()
    return coef[0].item(), coef[1].item(), r2.item()


def peak(points: list[dict]) -> dict:
    return max(points, key=lambda q: q["gold"])
