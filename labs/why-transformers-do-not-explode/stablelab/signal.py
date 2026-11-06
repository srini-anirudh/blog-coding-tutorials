"""Parts I and II of the post: variance through a matmul, deep plain stacks, mean-field signal propagation.

Everything here is a random network at initialization. No training happens in this file.
"""

from __future__ import annotations

import math

import numpy as np
import torch

ACTS = {
    "relu": torch.relu,
    "tanh": torch.tanh,
    "linear": lambda x: x,
}


def nonlinearity_gain(act: str, n: int = 2_000_000, seed: int = 0) -> float:
    """gamma = 1 / E[phi(z)^2] for z ~ N(0, 1), by Monte Carlo. Kaiming's rule is sigma_w^2 = gamma / n_in."""
    z = torch.randn(n, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)
    return 1.0 / ACTS[act](z).pow(2).mean().item()


def init_std(scheme: str, n_in: int, n_out: int) -> float:
    """Weight standard deviation for the three initializers the post races in Figure 02."""
    if scheme == "naive":
        return 0.01
    if scheme == "xavier":
        return math.sqrt(2.0 / (n_in + n_out))
    if scheme == "kaiming":
        return math.sqrt(2.0 / n_in)
    raise ValueError(scheme)


def plain_stack(depth: int, width: int, scheme: str, act: str, batch: int = 256, seed: int = 0) -> dict:
    """Forward and backward through `depth` plain (no residual, no norm) layers.

    Returns the second moment of the pre-activations at every layer (forward) and of the gradient with
    respect to every layer's input (backward), starting from a unit-variance gradient at the top.
    float64 so that the naive initializer's 1e-80 does not underflow to zero before we can print it.
    """
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, width, generator=g, dtype=torch.float64)
    std = init_std(scheme, width, width)
    weights = [torch.randn(width, width, generator=g, dtype=torch.float64) * std for _ in range(depth)]
    phi = ACTS[act]
    h = x.requires_grad_(True)
    inputs, fwd = [h], []
    for w in weights:
        z = h @ w.T
        fwd.append(z.pow(2).mean().item())
        h = phi(z)
        h.retain_grad()
        inputs.append(h)
    top = torch.randn(batch, width, generator=g, dtype=torch.float64)
    (h * top).sum().backward()
    bwd = [t.grad.pow(2).mean().item() for t in inputs]  # bwd[0] is the gradient at the network input
    return {"fwd": fwd, "bwd": bwd}


def log_variance_walk(depth: int, width: int, n_nets: int = 16, seed: int = 0) -> np.ndarray:
    """ln(||x_l||^2 / ||x_0||^2) for one input through `n_nets` critical linear networks.

    Each layer is W ~ N(0, 1/width), so the mean gain is exactly 1, but each layer's realized gain jitters.
    Returns an array of shape (n_nets, depth + 1).
    """
    g = torch.Generator().manual_seed(seed)
    out = np.zeros((n_nets, depth + 1))
    for i in range(n_nets):
        x = torch.randn(width, generator=g, dtype=torch.float64)
        n0 = x.pow(2).sum().item()
        for l in range(depth):
            x = torch.randn(width, width, generator=g, dtype=torch.float64) @ x / math.sqrt(width)
            out[i, l + 1] = math.log(x.pow(2).sum().item() / n0)
    return out


# --- Mean-field theory for a random tanh network (Poole et al. 2016, Schoenholz et al. 2016) ---------------

_Z, _W = np.polynomial.hermite_e.hermegauss(101)
_W = _W / _W.sum()  # Gauss-Hermite nodes and weights for E over z ~ N(0, 1)


def _e1(f) -> float:
    return float((_W * f(_Z)).sum())


def _e2(f, g, q: float, c: float) -> float:
    """E[f(u1) g(u2)] where (u1, u2) are Gaussian, variance q each, correlation c."""
    z1, z2 = np.meshgrid(_Z, _Z, indexing="ij")
    w = np.outer(_W, _W)
    u1 = math.sqrt(q) * z1
    u2 = math.sqrt(q) * (c * z1 + math.sqrt(max(1 - c * c, 0.0)) * z2)
    return float((w * f(u1) * g(u2)).sum())


def fixed_point_q(sw: float, sb: float, iters: int = 500) -> float:
    q = 1.0
    for _ in range(iters):
        q = sw**2 * _e1(lambda z: np.tanh(math.sqrt(q) * z) ** 2) + sb**2
    return q


def correlation_map(c: float, sw: float, sb: float, q: float) -> float:
    """C(c): correlation of two inputs one layer up, given correlation c and length q* below."""
    return (sw**2 * _e2(np.tanh, np.tanh, q, c) + sb**2) / q


def chi1(sw: float, sb: float) -> float:
    """chi_1 = sigma_w^2 E[phi'(z)^2] at the fixed point q*. Equals the per-layer backward gain."""
    q = fixed_point_q(sw, sb)
    return sw**2 * _e1(lambda z: 1 / np.cosh(math.sqrt(q) * z) ** 4)


def correlation_fixed_point(sw: float, sb: float, iters: int = 2000) -> tuple[float, float]:
    """Returns (c*, slope of C at c*). Depth scale is -1 / ln(slope)."""
    q = fixed_point_q(sw, sb)
    c = 0.5
    for _ in range(iters):
        c = min(correlation_map(c, sw, sb, q), 1.0)
    h = 1e-4
    lo = min(c, 1 - h) - h
    slope = (correlation_map(lo + h, sw, sb, q) - correlation_map(lo, sw, sb, q)) / h
    return c, slope


def critical_sigma_w(sb: float, lo: float = 0.5, hi: float = 4.0) -> float:
    """Bisection on chi_1(sigma_w) = 1: the order-to-chaos boundary for a given bias std."""
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if chi1(mid, sb) < 1 else (lo, mid)
    return (lo + hi) / 2


def empirical_correlations(sw: float, sb: float, depth: int, width: int = 1000, c0: float = 0.2,
                           seed: int = 0) -> list[float]:
    """Push two inputs with correlation c0 through one random tanh net; correlation of pre-activations per layer."""
    g = torch.Generator().manual_seed(seed)
    q0 = fixed_point_q(sw, sb)
    a = torch.randn(width, generator=g, dtype=torch.float64)
    b = c0 * a + math.sqrt(1 - c0**2) * torch.randn(width, generator=g, dtype=torch.float64)
    h = torch.stack([a, b]) * math.sqrt(q0)
    out = []
    for _ in range(depth):
        w = torch.randn(width, width, generator=g, dtype=torch.float64) * sw / math.sqrt(width)
        bias = torch.randn(width, generator=g, dtype=torch.float64) * sb
        h = torch.tanh(h) @ w.T + bias
        out.append((h[0] @ h[1] / h[0].norm() / h[1].norm()).item())  # q12 / sqrt(q11 q22)
    return out


# --- Dynamical isometry: the singular values of the end-to-end Jacobian --------------------------------------

def critical_tanh_point(q_star: float) -> tuple[float, float]:
    """(sigma_w, sigma_b) on the tanh critical line with the given fixed-point length q*."""
    sw2 = 1.0 / _e1(lambda z: 1 / np.cosh(math.sqrt(q_star) * z) ** 4)
    sb2 = q_star - sw2 * _e1(lambda z: np.tanh(math.sqrt(q_star) * z) ** 2)
    return math.sqrt(sw2), math.sqrt(max(sb2, 0.0))


def jacobian_singular_values(depth: int, width: int, act: str, weights: str, sw: float, sb: float,
                             seed: int = 0) -> np.ndarray:
    """Singular values of J = d x_L / d x_0 for one random network, built as an explicit product."""
    g = torch.Generator().manual_seed(seed)
    q = fixed_point_q(sw, sb) if act == "tanh" else 1.0
    h = torch.randn(width, generator=g, dtype=torch.float64) * math.sqrt(q)
    jac = torch.eye(width, dtype=torch.float64)
    for _ in range(depth):
        if weights == "orthogonal":
            w = torch.linalg.qr(torch.randn(width, width, generator=g, dtype=torch.float64))[0] * sw
        else:
            w = torch.randn(width, width, generator=g, dtype=torch.float64) * sw / math.sqrt(width)
        bias = torch.randn(width, generator=g, dtype=torch.float64) * sb
        if act == "tanh":
            d = 1 - torch.tanh(h) ** 2
            h = torch.tanh(h) @ w.T + bias
        else:  # relu
            d = (h > 0).double()
            h = torch.relu(h) @ w.T + bias
        jac = (w * d) @ jac  # W diag(phi'(h)) J
    return torch.linalg.svdvals(jac).numpy()
