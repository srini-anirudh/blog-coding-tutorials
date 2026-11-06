"""Additive attention biases. Each returns a float tensor added to the logits.

A mask is the special case of a bias that only takes the values 0 and -inf.
All biases here broadcast to (B, H, Nq, Nk), so they compose by addition.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .masks import causal_mask, mask_to_bias, sliding_window_mask


def causal_bias(nq: int, nk: int, device: torch.device | str | None = None) -> torch.Tensor:
    """0 on and below the (bottom-right) diagonal, -inf above. Shape (nq, nk)."""
    return mask_to_bias(causal_mask(nq, nk, device=device))


def window_bias(nq: int, nk: int, window: int, device: torch.device | str | None = None) -> torch.Tensor:
    """0 inside the causal sliding window, -inf outside. Shape (nq, nk)."""
    return mask_to_bias(sliding_window_mask(nq, window, nk, device=device))


def alibi_slopes(num_heads: int) -> torch.Tensor:
    """ALiBi head slopes: a geometric sequence 2^(-8/H), 2^(-16/H), ... (Press et al., 2022).

    For a non-power-of-two head count, take the slopes of the nearest lower power
    of two and fill the rest with every other slope of the next power, as the paper does.
    """

    def power_of_two(n: int) -> list[float]:
        start = 2 ** (-8 / n)
        return [start ** (i + 1) for i in range(n)]

    if math.log2(num_heads).is_integer():
        return torch.tensor(power_of_two(num_heads))
    lower = 2 ** math.floor(math.log2(num_heads))
    extra = power_of_two(2 * lower)[0::2][: num_heads - lower]
    return torch.tensor(power_of_two(lower) + extra)


def alibi_bias(slopes: torch.Tensor, nq: int, nk: int) -> torch.Tensor:
    """-slope_h * |distance| for every query-key pair. Returns (1, H, nq, nk).

    Nearby keys are barely penalised; far keys are pushed down linearly. Steep heads
    look locally, shallow heads look far. Queries are the last nq positions.
    """
    q_pos = torch.arange(nq, device=slopes.device)[:, None] + (nk - nq)
    k_pos = torch.arange(nk, device=slopes.device)[None, :]
    distance = (q_pos - k_pos).abs().float()
    return (-slopes[:, None, None] * distance)[None]


class RelativePositionBias(nn.Module):
    """A learned scalar per head for each clipped relative distance (T5/Swin style, 1D).

    table: (2 * max_distance + 1, H). Distances beyond max_distance share one entry.
    """

    def __init__(self, num_heads: int, max_distance: int = 64) -> None:
        super().__init__()
        self.max_distance = max_distance
        self.table = nn.Parameter(torch.zeros(2 * max_distance + 1, num_heads))

    def forward(self, nq: int, nk: int) -> torch.Tensor:
        """Returns (1, H, nq, nk)."""
        q_pos = torch.arange(nq, device=self.table.device)[:, None] + (nk - nq)
        k_pos = torch.arange(nk, device=self.table.device)[None, :]
        rel = (k_pos - q_pos).clamp(-self.max_distance, self.max_distance) + self.max_distance
        return self.table[rel].permute(2, 0, 1)[None]
