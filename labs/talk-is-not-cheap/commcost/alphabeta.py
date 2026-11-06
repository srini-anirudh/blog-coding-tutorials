"""The alpha-beta cost model and the closed-form cost of each AllReduce algorithm in the post.

T(n) = alpha + beta * n. alpha is seconds per message, beta is seconds per byte (one over bandwidth).
"""

from __future__ import annotations

import math

import numpy as np


def message_time(n: float, alpha: float, beta: float) -> float:
    return alpha + beta * n


def n_half(alpha: float, beta: float) -> float:
    """Message size at which fixed and per-byte costs are equal: you get half the peak bandwidth."""
    return alpha / beta


def effective_bandwidth(n: float, alpha: float, beta: float) -> float:
    return n / message_time(n, alpha, beta)


def fit_alpha_beta(sizes, times) -> tuple[float, float]:
    """Least-squares fit of T = alpha + beta n, weighted by 1/T so small and large messages count equally.

    An unweighted fit lets the largest message decide everything and leaves alpha as noise.
    """
    n = np.asarray(sizes, dtype=float)
    t = np.asarray(times, dtype=float)
    w = 1.0 / t
    a = np.stack([np.ones_like(n), n], axis=1) * w[:, None]
    (alpha, beta), *_ = np.linalg.lstsq(a, t * w, rcond=None)
    return float(alpha), float(beta)


# Closed forms. n is the vector size in bytes on every rank, p the number of ranks. gamma is seconds per byte reduced.
def t_naive(n, p, alpha, beta):
    """Everyone sends to rank 0, rank 0 sends the sum back. One port, so 2(p-1) full-vector messages in a row."""
    return 2 * (p - 1) * (alpha + n * beta)


def t_tree(n, p, alpha, beta):
    """Binomial reduce then binomial broadcast, full vector each round."""
    return 2 * math.ceil(math.log2(p)) * (alpha + n * beta)


def t_ring(n, p, alpha, beta, gamma=0.0):
    return 2 * (p - 1) * alpha + 2 * (p - 1) / p * n * beta + (p - 1) / p * n * gamma


def t_rhd(n, p, alpha, beta, gamma=0.0):
    """Rabenseifner: recursive-halving reduce-scatter then recursive-doubling all-gather (p a power of two)."""
    return 2 * math.log2(p) * alpha + 2 * (p - 1) / p * n * beta + (p - 1) / p * n * gamma


def t_hierarchical(n, g, nodes, alpha_nv, beta_nv, alpha_net, beta_net):
    """Ring reduce-scatter inside the node, ring AllReduce of each 1/G slice across nodes, ring all-gather inside."""
    intra = 2 * (g - 1) * alpha_nv + 2 * (g - 1) / g * n * beta_nv
    inter = 2 * (nodes - 1) * alpha_net + 2 * (nodes - 1) / nodes * (n / g) * beta_net
    return intra + inter


def t_in_network(n, alpha, beta):
    """Switch-side reduction (SHARP / NVLS): the vector goes up once and the sum comes down once."""
    return 2 * alpha + 2 * n * beta


def bytes_per_rank(algorithm: str, n: float, p: int) -> float:
    """Bytes each non-root rank sends. For naive and tree the root is the bottleneck, so return the root's count."""
    if algorithm == "naive":
        return (p - 1) * n
    if algorithm == "tree":
        return math.ceil(math.log2(p)) * n
    if algorithm in ("ring", "rhd"):
        return 2 * (p - 1) / p * n
    if algorithm == "in_network":
        return n
    raise ValueError(algorithm)


def allreduce_lower_bound_bytes(n: float, p: int) -> float:
    """Each rank must send at least 2(p-1)/p n bytes (reduce-scatter bound plus all-gather bound)."""
    return 2 * (p - 1) / p * n


def crossover_size(f, g, p, alpha, beta, lo=1.0, hi=1e11) -> float | None:
    """Smallest n where algorithm g becomes faster than f, by bisection in log space. None if no crossover."""
    d = lambda n: f(n, p, alpha, beta) - g(n, p, alpha, beta)  # noqa: E731
    if d(lo) * d(hi) > 0:
        return None
    a, b = math.log(lo), math.log(hi)
    for _ in range(200):
        m = (a + b) / 2
        if d(math.exp(a)) * d(math.exp(m)) <= 0:
            b = m
        else:
            a = m
    return math.exp(b)


def bus_bandwidth(n_bytes: float, seconds: float, p: int) -> float:
    """nccl-tests convention for AllReduce: algorithm bandwidth times 2(p-1)/p."""
    return n_bytes / seconds * 2 * (p - 1) / p
