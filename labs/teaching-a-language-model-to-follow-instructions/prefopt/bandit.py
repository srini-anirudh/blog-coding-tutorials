"""A one-prompt "language model" over a fixed list of N completions, small enough to compute every probability.

The policy shares parameters across completions, as a real network does:

    log pi_theta(y) = log_softmax( log ref(y) + phi(y) . theta )

phi(y) is a fixed d-dimensional feature (think: the final hidden state). theta = 0 gives pi = ref. Because
completions share features, pushing one pair apart moves probability on completions that never appear in the data.
That coupling is what drives likelihood displacement and off-support drift.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .losses import dpo_sequence_loss
from .tilt import kl


@dataclass
class World:
    phi: torch.Tensor  # (N, d) features
    ref_logp: torch.Tensor  # (N,) reference log-probs


def unit_rows(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True)


def policy_logp(world: World, theta: torch.Tensor) -> torch.Tensor:
    return torch.log_softmax(world.ref_logp + world.phi @ theta, dim=-1)


def similar_pairs_world(n_pairs: int, n_other: int, dim: int, similarity: float, seed: int = 0):
    """n_pairs (chosen, rejected) pairs whose features have cosine `similarity`, plus n_other unrelated completions.

    Rows [0, P) are chosen, rows [P, 2P) are rejected, the rest never appear in the data. The reference is
    uniform, so every effect below comes from the shared features alone.
    """
    gen = torch.Generator().manual_seed(seed)
    w = unit_rows(torch.randn(n_pairs, dim, generator=gen, dtype=torch.float64))
    z = torch.randn(n_pairs, dim, generator=gen, dtype=torch.float64)
    z = unit_rows(z - (z * w).sum(-1, keepdim=True) * w)  # orthogonal to w
    lose = similarity * w + (1 - similarity**2) ** 0.5 * z  # cosine(w, lose) = similarity exactly
    other = unit_rows(torch.randn(n_other, dim, generator=gen, dtype=torch.float64))
    phi = torch.cat([w, lose, other])
    n = phi.shape[0]
    world = World(phi, torch.full((n,), -torch.log(torch.tensor(float(n), dtype=torch.float64))))
    chosen = torch.arange(n_pairs)
    return world, chosen, chosen + n_pairs


def train_dpo(world: World, chosen: torch.Tensor, rejected: torch.Tensor, beta: float = 0.1, steps: int = 400,
              lr: float = 0.05, dpop_lambda: float = 0.0, nll_alpha: float = 0.0, log_every: int = 50,
              theta: torch.Tensor | None = None):
    """Full-batch DPO with Adam on theta. Returns theta and a trajectory of what happened to the probabilities."""
    theta = torch.zeros(world.phi.shape[1], dtype=torch.float64) if theta is None else theta.clone()
    theta.requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=lr)
    ref_w, ref_l = world.ref_logp[chosen], world.ref_logp[rejected]
    in_data = torch.zeros(world.phi.shape[0], dtype=torch.bool)
    in_data[chosen] = True
    in_data[rejected] = True
    history = []
    for step in range(steps + 1):
        logp = policy_logp(world, theta)
        loss = dpo_sequence_loss(logp[chosen], logp[rejected], ref_w, ref_l, beta, dpop_lambda, nll_alpha)
        if step % log_every == 0 or step == steps:
            with torch.no_grad():
                p = logp.exp()
                history.append({
                    "step": step, "loss": loss.item(),
                    "d_logp_chosen": (logp[chosen] - ref_w).mean().item(),
                    "d_logp_rejected": (logp[rejected] - ref_l).mean().item(),
                    "mass_chosen": p[chosen].sum().item(), "mass_rejected": p[rejected].sum().item(),
                    "mass_elsewhere": p[~in_data].sum().item(),
                    "kl": kl(logp, world.ref_logp).item(),
                })
        if step == steps:
            break
        opt.zero_grad()
        loss.backward()
        opt.step()
    return theta.detach(), history


def bt_margin_grad(world: World, theta: torch.Tensor, w: torch.Tensor, l: torch.Tensor, beta: float):
    """d loss / d theta for one batch, for tests that check the DPO gradient formula."""
    t = theta.clone().requires_grad_(True)
    logp = policy_logp(world, t)
    loss = -F.logsigmoid(beta * ((logp[w] - world.ref_logp[w]) - (logp[l] - world.ref_logp[l]))).mean()
    (g,) = torch.autograd.grad(loss, t)
    return g
