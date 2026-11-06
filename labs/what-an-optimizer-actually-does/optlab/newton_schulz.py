"""Orthogonalize a matrix without an SVD, and the matrix-aware directions around it.

Muon's quintic Newton-Schulz step is X <- a X + b (X X^T) X + c (X X^T)^2 X.
X and X X^T share singular vectors, so on each singular value it is the scalar map
s <- a s + b s^3 + c s^5. That is why it can flatten the spectrum without computing it.
"""

from __future__ import annotations

import torch

MUON_COEFFS = (3.4445, -4.7750, 2.0315)  # tuned for a steep slope at 0 (Keller Jordan's Muon)
CUBIC_COEFFS = (1.5, -0.5, 0.0)  # the textbook iteration, which converges to exactly 1


def newton_schulz(g: torch.Tensor, steps: int = 5, coeffs=MUON_COEFFS, dtype=torch.bfloat16) -> torch.Tensor:
    """Approximate U V^T for g = U S V^T. Runs in `dtype` (bf16 by default, as in Muon)."""
    a, b, c = coeffs
    x = g.to(dtype)
    tall = x.shape[0] > x.shape[1]
    if tall:  # work on the wide side so X X^T is the smaller Gram matrix
        x = x.T
    x = x / (x.norm() + 1e-7)  # Frobenius norm >= spectral norm, so every singular value <= 1
    for _ in range(steps):
        A = x @ x.T
        B = b * A + c * A @ A
        x = a * x + B @ x
    if tall:
        x = x.T
    return x.to(g.dtype)


def ns_scalar(s: torch.Tensor, steps: int = 5, coeffs=MUON_COEFFS) -> torch.Tensor:
    """The same iteration applied to singular values directly."""
    a, b, c = coeffs
    for _ in range(steps):
        s = a * s + b * s**3 + c * s**5
    return s


def spectral_steepest(g: torch.Tensor) -> torch.Tensor:
    """argmin <G, D> subject to sigma_max(D) <= 1, which is -U V^T (exact, via SVD)."""
    u, _, vh = torch.linalg.svd(g, full_matrices=False)
    return -(u @ vh)


def matrix_power(sym: torch.Tensor, power: float, eps: float = 1e-12) -> torch.Tensor:
    """sym^power for a symmetric positive semi-definite matrix, by eigendecomposition."""
    w, q = torch.linalg.eigh(sym)
    return q @ torch.diag(w.clamp_min(eps) ** power) @ q.T


def shampoo_direction(L: torch.Tensor, R: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """Shampoo's step direction L^-1/4 G R^-1/4."""
    return matrix_power(L, -0.25) @ g @ matrix_power(R, -0.25)
