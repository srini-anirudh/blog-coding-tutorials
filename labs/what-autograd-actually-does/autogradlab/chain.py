"""Forward mode against reverse mode: the matrix-chain view and a timed version on a real network.

The derivative of a chain of layers is a product of Jacobians J_k ... J_1. Its value does not depend on the
parenthesisation. Its cost does. Grouping from the output end is reverse mode; grouping from the input end is
forward mode.
"""

from __future__ import annotations

import time

import torch
from torch import nn
from torch.func import functional_call, jacfwd, jacrev, jvp, vjp


def chain_cost(widths: list[int]) -> dict[str, int]:
    """Multiply-adds to form J_k ... J_1 for dense Jacobians, J_i of shape widths[i] x widths[i-1].

    widths = [n_0, n_1, ..., n_k]; n_k is the output size. Reverse multiplies from J_k down; forward from J_1 up.
    Also returns the largest intermediate product each order holds.
    """
    k = len(widths) - 1
    # Reverse: running product R has shape n_k x n_i. R <- R @ J_i costs n_k * n_i * n_{i-1}.
    reverse = sum(widths[k] * widths[i] * widths[i - 1] for i in range(k - 1, 0, -1)) if k > 1 else 0
    # Forward: running product F has shape n_i x n_0. F <- J_{i+1} @ F costs n_{i+1} * n_i * n_0.
    forward = sum(widths[i + 1] * widths[i] * widths[0] for i in range(1, k)) if k > 1 else 0
    peak_rev = max(widths[k] * widths[i] for i in range(0, k))
    peak_fwd = max(widths[i] * widths[0] for i in range(1, k + 1))
    return dict(reverse=reverse, forward=forward, peak_reverse=peak_rev, peak_forward=peak_fwd)


def random_jacobians(widths: list[int], seed: int = 0) -> list[torch.Tensor]:
    """Dense random J_i of shape widths[i] x widths[i-1], in float64."""
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(widths[i], widths[i - 1], generator=g, dtype=torch.float64) for i in range(1, len(widths))]


def multiply_chain(J: list[torch.Tensor], order: str) -> torch.Tensor:
    """J_k ... J_1, multiplied from the output end ("reverse") or from the input end ("forward")."""
    if order == "reverse":
        out = J[-1]
        for Ji in reversed(J[:-1]):
            out = out @ Ji
    else:
        out = J[0]
        for Ji in J[1:]:
            out = Ji @ out
    return out


def mlp(n_in: int, width: int, depth: int, seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    layers = [nn.Linear(n_in, width), nn.Tanh()]
    for _ in range(depth - 1):
        layers += [nn.Linear(width, width), nn.Tanh()]
    layers += [nn.Linear(width, 1)]
    return nn.Sequential(*layers)


def gradient_three_ways(model: nn.Module, x: torch.Tensor) -> dict[str, dict]:
    """d(scalar output)/d(input) by reverse mode, forward mode and central finite differences.

    Each entry has the gradient, the wall time and how many sweeps (forward-equivalent passes) were needed.
    """
    f = lambda inp: model(inp).sum()  # noqa: E731
    out = {}

    t = time.perf_counter()
    g_rev = jacrev(f)(x)
    out["reverse"] = dict(grad=g_rev, seconds=time.perf_counter() - t, sweeps=1)

    t = time.perf_counter()
    g_fwd = jacfwd(f, randomness="same")(x)  # vmapped JVPs, one basis direction per input
    out["forward"] = dict(grad=g_fwd, seconds=time.perf_counter() - t, sweeps=x.numel())

    t = time.perf_counter()
    eps = 1e-3 if x.dtype == torch.float64 else 1e-2
    flat = x.reshape(-1)
    g_fd = torch.empty_like(flat)
    with torch.no_grad():
        for i in range(flat.numel()):
            e = torch.zeros_like(flat)
            e[i] = eps
            g_fd[i] = (f((flat + e).view_as(x)) - f((flat - e).view_as(x))) / (2 * eps)
    out["finite_diff"] = dict(grad=g_fd.view_as(x), seconds=time.perf_counter() - t, sweeps=2 * x.numel())
    return out


def jvp_vjp_dot_test(f, x: torch.Tensor, seed: int = 0) -> tuple[float, float]:
    """<u, J v> and <J^T u, v> for random u, v. They agree if the VJP is the transpose of the JVP."""
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(x.shape, generator=g, dtype=x.dtype)
    y, Jv = jvp(f, (x,), (v,))
    u = torch.randn(y.shape, generator=g, dtype=x.dtype)
    _, pullback = vjp(f, x)
    (JTu,) = pullback(u)
    return float((u * Jv).sum()), float((JTu * v).sum())


def param_gradient_reverse(model: nn.Module, x: torch.Tensor) -> int:
    """Number of parameters that one reverse sweep gives a gradient for."""
    params = dict(model.named_parameters())
    grads = torch.func.grad(lambda p: functional_call(model, p, (x,)).sum())(params)
    return sum(g.numel() for g in grads.values())
