"""Per-task gradients, the gradient Gram matrix K, and PCGrad (post section 3).

The first-order law: a step of size eta along g = sum_i p_i g_i changes task j's loss by about
-eta * sum_i p_i <g_i, g_j> = -eta * (K p)_j. `one_step_check` takes that step for real and compares.
"""

from __future__ import annotations

import copy
from typing import Mapping, Sequence

import torch

from .tasks import Pool
from .train import mixture_loss


def task_loss(model, pool: Pool, n: int | None = None) -> torch.Tensor:
    """Token-mean loss on the first n examples of a pool (deterministic, so repeated calls agree exactly)."""
    n = n or len(pool)
    return mixture_loss(model, pool.input_ids[:n], pool.labels[:n], norm="token")


def flat_grad(model, loss: torch.Tensor) -> torch.Tensor:
    params = [p for p in model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(loss, params)
    return torch.cat([g.reshape(-1) for g in grads])


def task_gradients(model, pools: Mapping[str, Pool], n: int = 1024) -> dict[str, torch.Tensor]:
    return {name: flat_grad(model, task_loss(model, pool, n)) for name, pool in pools.items()}


def gram(grads: Sequence[torch.Tensor]) -> torch.Tensor:
    """K_ij = <g_i, g_j>. Symmetric and positive semi-definite by construction."""
    g = torch.stack(list(grads))
    return g @ g.T


def cosine(grads: Sequence[torch.Tensor]) -> torch.Tensor:
    g = torch.stack(list(grads))
    g = g / g.norm(dim=1, keepdim=True)
    return g @ g.T


def pcgrad_pair(ga: torch.Tensor, gb: torch.Tensor) -> torch.Tensor:
    """Project g_A onto the normal plane of g_B when they conflict (Yu et al., 2020); otherwise keep g_A."""
    dot = torch.dot(ga, gb)
    if dot >= 0:
        return ga
    return ga - dot / gb.dot(gb) * gb


def pcgrad(grads: Sequence[torch.Tensor], order: Sequence[int] | None = None) -> torch.Tensor:
    """Full PCGrad: each task gradient is projected against every other task it conflicts with, then summed."""
    out = []
    for i, gi in enumerate(grads):
        g = gi.clone()
        for j in order if order is not None else range(len(grads)):
            if j != i:
                g = pcgrad_pair(g, grads[j])
        out.append(g)
    return torch.stack(out).sum(0)


@torch.no_grad()
def _apply(model, direction: torch.Tensor, eta: float):
    moved = copy.deepcopy(model)
    offset = 0
    for p in moved.parameters():
        n = p.numel()
        p -= eta * direction[offset : offset + n].view_as(p)
        offset += n
    return moved


def one_step_check(model, pools: Mapping[str, Pool], direction: torch.Tensor, eta: float, n: int = 1024):
    """Actual change in every task's loss after one SGD step along `direction`, and the first-order prediction.

    Returns {task: (actual, predicted)} where predicted = -eta * <direction, g_task>.
    """
    grads = task_gradients(model, pools, n)
    moved = _apply(model, direction, eta)
    out = {}
    with torch.no_grad():
        for name, pool in pools.items():
            before = float(task_loss(model, pool, n))
            after = float(task_loss(moved, pool, n))
            out[name] = (after - before, float(-eta * direction.dot(grads[name])))
    return out
