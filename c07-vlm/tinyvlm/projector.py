"""The projector: vision features (d_v) -> language-model embeddings (d_lm).

Shapes, with n = number of crops, N_v = patch tokens per crop:
    input   n x N_v x d_v          SigLIP-base: n x 196 x 768
    pixel shuffle (factor s)       n x (N_v / s^2) x (d_v * s^2), e.g. n x 49 x 3072
    output  n x N_img x d_lm       SmolLM2-135M: d_lm = 576
"""

from __future__ import annotations

import math

import torch
from torch import nn


class LinearProjector(nn.Module):
    """The smallest possible bridge: one affine map per patch token (LLaVA-1)."""

    def __init__(self, d_in: int, d_lm: int):
        super().__init__()
        self.proj = nn.Linear(d_in, d_lm)

    def forward(self, features: torch.Tensor) -> torch.Tensor:  # n x N x d_in -> n x N x d_lm
        return self.proj(features)


class MLPProjector(nn.Module):
    """Linear -> GELU -> Linear (LLaVA-1.5). The nonlinearity lets it bend the feature space."""

    def __init__(self, d_in: int, d_lm: int):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_lm)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(d_lm, d_lm)

    def forward(self, features: torch.Tensor) -> torch.Tensor:  # n x N x d_in -> n x N x d_lm
        return self.fc2(self.act(self.fc1(features)))


def pixel_shuffle(features: torch.Tensor, factor: int) -> torch.Tensor:
    """Fold each factor x factor block of neighbouring patches into one token (SmolVLM, Idefics3).

    n x N x d  ->  n x (N / factor^2) x (d * factor^2). No information is dropped: the 4 patches of
    a 2 x 2 block are concatenated along the channel axis, so the token count shrinks 4x.
    """
    if factor == 1:
        return features
    n, num_tokens, dim = features.shape
    side = math.isqrt(num_tokens)
    if side * side != num_tokens or side % factor:
        raise ValueError(f"{num_tokens} tokens do not form a square grid divisible by {factor}")
    x = features.reshape(n, side, side // factor, dim * factor)  # merge `factor` columns
    x = x.permute(0, 2, 1, 3)  # n x (side/f) x side x (d*f)
    x = x.reshape(n, side // factor, side // factor, dim * factor * factor)  # merge `factor` rows
    x = x.permute(0, 2, 1, 3)  # back to row-major order of the coarse grid
    return x.reshape(n, (side // factor) ** 2, dim * factor * factor)


def build_projector(projector_type: str, d_in: int, d_lm: int) -> nn.Module:
    if projector_type == "linear":
        return LinearProjector(d_in, d_lm)
    if projector_type == "mlp":
        return MLPProjector(d_in, d_lm)
    raise ValueError(f"unknown projector_type {projector_type!r}")
