"""Behaviour cloning, compounding error and DAgger on a toy where the expert's states cover only a thin band.

World: a point in the plane that should circle a unit ring at speed 0.1 per step. The expert's action moves
along the ring and pulls the point back to it. Process noise is small, so the states the expert visits lie
in a thin band around the ring. A learner (a small MLP) is fit to expert (state, action) pairs by least
squares. Cost per step is 1 if the point is more than 0.1 from the ring and 0 otherwise (bounded, as in the classical analysis). Nothing in the expert data shows what to do
far from the ring, so a learner that drifts out gets no useful signal there.

This is a stand-in for the post's argument, not a language model. The shape of the cost against the horizon
is measured, not assumed.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

SPEED, GAIN = 0.1, 0.5
BAND = 0.1   # a step costs 1 when the point is more than BAND away from the ring, else 0


def expert_action(s: torch.Tensor) -> torch.Tensor:
    r = s.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    radial = s / r
    tangent = torch.stack([-radial[..., 1], radial[..., 0]], -1)
    return SPEED * tangent + GAIN * (radial - s)


def rollout(policy, n: int, T: int, noise: float, gen: torch.Generator, action_noise: float = 0.0,
            init_noise: float = 0.01, beta: float = 0.0, expert_mix_gen=None):
    """Roll out `n` trajectories. Returns states (n, T, 2) and per-step cost (n, T).
    action_noise: Gaussian noise added to the executed action (labels stay clean) - the DART trick.
    beta: probability of executing the expert's action instead of the policy's at each step (DAgger mixing)."""
    th = torch.rand(n, generator=gen) * 2 * math.pi
    s = torch.stack([th.cos(), th.sin()], -1) + init_noise * torch.randn(n, 2, generator=gen)
    S, C = [], []
    for _ in range(T):
        S.append(s)
        C.append(((s.norm(dim=-1) - 1).abs() > BAND).float())
        with torch.no_grad():
            a = policy(s)
            if beta > 0:
                use_exp = (torch.rand(n, 1, generator=gen) < beta).float()
                a = use_exp * expert_action(s) + (1 - use_exp) * a
        if action_noise > 0:
            a = a + action_noise * torch.randn(n, 2, generator=gen)
        s = s + a + noise * torch.randn(n, 2, generator=gen)
    return torch.stack(S, 1), torch.stack(C, 1)


def make_mlp(hidden: int = 64) -> nn.Module:
    return nn.Sequential(nn.Linear(2, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 2))


def fit(states: torch.Tensor, steps: int = 1500, seed: int = 0, lr: float = 3e-3, hidden: int = 64) -> nn.Module:
    torch.manual_seed(seed)
    net = make_mlp(hidden)
    X = states.reshape(-1, 2)
    Y = expert_action(X)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for i in range(steps):
        idx = torch.randint(0, len(X), (512,))
        loss = ((net(X[idx]) - Y[idx]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return net


def mean_cost(policy, T: int, noise: float, seed: int, n: int = 200) -> float:
    gen = torch.Generator().manual_seed(10_000 + seed)
    _, C = rollout(policy, n, T, noise, gen)
    return float(C.sum(1).mean())


def loglog_slope(Ts, ys) -> float:
    return float(np.polyfit(np.log(Ts), np.log(ys), 1)[0])


def tabular_worst_case(eps: float, T: int, n: int = 20000, seed: int = 0, recover: bool = False) -> float:
    """The classical construction: on the expert's states the learner errs with probability eps; one error
    leaves the covered region and, if there is no recovery data, costs 1 at every later step. With recovery data
    (`recover`, the DAgger case) the cost of an error is only the step on which it happens."""
    rng = np.random.default_rng(seed)
    err = rng.random((n, T)) < eps
    first = np.where(err.any(1), err.argmax(1), T)
    if recover:
        return float(err.sum(1).mean())
    return float(np.mean(T - first))
