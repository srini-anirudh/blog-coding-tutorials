"""Rank as a property of the singular value spectrum (the post's Figure 2), and what LoRA does instead of truncating.

Eckart-Young: the best rank-r approximation in Frobenius norm keeps the top r singular triplets, and its error is
sqrt(sum_{i>r} sigma_i^2). LoRA never sees the full matrix. It searches the rank-r set with gradient descent.
"""

from __future__ import annotations

import torch


def matrix_with_spectrum(sigmas: torch.Tensor, m: int, n: int, seed: int = 0) -> torch.Tensor:
    """A random m x n matrix whose singular values are exactly `sigmas` (random orthonormal U and V)."""
    gen = torch.Generator().manual_seed(seed)
    k = len(sigmas)
    U, _ = torch.linalg.qr(torch.randn(m, k, generator=gen, dtype=torch.float64))
    V, _ = torch.linalg.qr(torch.randn(n, k, generator=gen, dtype=torch.float64))
    return U @ torch.diag(sigmas.double()) @ V.T


def decay_spectrum(k: int, kind: str) -> torch.Tensor:
    """'fast': geometric decay, a few patterns carry everything. 'slow': nearly flat."""
    i = torch.arange(k, dtype=torch.float64)
    if kind == "fast":
        return 10.0 * 0.6 ** i
    if kind == "slow":
        return 10.0 * (1 - 0.02 * i)
    raise ValueError(kind)


def truncate(M: torch.Tensor, r: int) -> torch.Tensor:
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    return U[:, :r] @ torch.diag(S[:r]) @ Vh[:r]


def eckart_young_error(M: torch.Tensor, r: int) -> float:
    S = torch.linalg.svdvals(M)
    return S[r:].pow(2).sum().sqrt().item()


def fit_factored(M: torch.Tensor, r: int, steps: int = 3000, lr: float = 1e-2, seed: int = 0) -> tuple[float, list[float]]:
    """Find B A of rank r by Adam on ||M - B A||_F^2, starting LoRA-style from B = 0, A random.

    Returns the final Frobenius error and the error curve. This is the LoRA search, with no SVD anywhere.
    """
    gen = torch.Generator().manual_seed(seed)
    m, n = M.shape
    M = M.float()
    A = (torch.randn(r, n, generator=gen) / n ** 0.5).requires_grad_()
    B = torch.zeros(m, r, requires_grad=True)
    opt = torch.optim.Adam([A, B], lr=lr)
    curve = []
    for step in range(steps):
        err = (M - B @ A).pow(2).sum()
        opt.zero_grad()
        err.backward()
        opt.step()
        if step % 100 == 0 or step == steps - 1:
            curve.append(err.sqrt().item())
    with torch.no_grad():
        final = (M - B @ A).norm().item()
    return final, curve
