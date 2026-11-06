"""The sinusoidal code of Vaswani et al. (2017), laid out as (sin, cos) pairs.

PE(p, 2i) = sin(p * w_i), PE(p, 2i+1) = cos(p * w_i), with w_i = 1 / 10000^(2i/d).
"""

from __future__ import annotations

import torch


def frequencies(dim: int, base: float = 10_000.0) -> torch.Tensor:
    """(dim/2,) angular speeds w_i. Pair 0 has wavelength 2*pi; the last pair is the slowest."""
    return 1.0 / base ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim)


def sinusoidal(positions: torch.Tensor, dim: int, base: float = 10_000.0) -> torch.Tensor:
    """(N,) positions -> (N, dim) with interleaved [sin, cos] per pair."""
    angles = positions.double()[:, None] * frequencies(dim, base)[None]
    return torch.stack([angles.sin(), angles.cos()], dim=-1).flatten(1)


def shift_matrix(k: float, dim: int, base: float = 10_000.0) -> torch.Tensor:
    """The fixed block-diagonal R_k with PE(p + k) = R_k @ PE(p) for every p.

    Each 2x2 block maps (sin a, cos a) to (sin(a + b), cos(a + b)) with b = k * w_i.
    """
    b = k * frequencies(dim, base)
    R = torch.zeros(dim, dim, dtype=torch.float64)
    for i, bi in enumerate(b):
        c, s = torch.cos(bi), torch.sin(bi)
        # sin(a+b) = sin a cos b + cos a sin b ; cos(a+b) = cos a cos b - sin a sin b
        R[2 * i : 2 * i + 2, 2 * i : 2 * i + 2] = torch.tensor([[c, s], [-s, c]], dtype=torch.float64)
    return R
