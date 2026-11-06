"""Scaled dot-product attention, written out by hand.

This is the reference everything else in the package is checked against.
It favours readability over speed: every intermediate tensor is materialised.
"""

from __future__ import annotations

import math

import torch
from torch import nn


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Attention with every step visible.

    Shapes (any number of leading batch dims, usually B x H):
        q:    (..., Nq, d)
        k:    (..., Nk, d)
        v:    (..., Nk, dv)
        mask: bool, broadcastable to (..., Nq, Nk). True means "may attend".
    Returns:
        out:     (..., Nq, dv)
        weights: (..., Nq, Nk), each row sums to 1
    """
    d = q.size(-1)
    scale = 1.0 / math.sqrt(d) if scale is None else scale

    scores = q @ k.transpose(-2, -1)  # (..., Nq, Nk): one dot product per query-key pair
    scores = scores * scale  # keeps the logit variance near 1 when d grows
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))  # exp(-inf) = 0 after softmax

    weights = torch.softmax(scores, dim=-1)  # normalise over keys
    out = weights @ v  # (..., Nq, dv): a weighted average of value rows
    return out, weights


class SelfAttentionHead(nn.Module):
    """One attention head: project the same sequence into queries, keys and values.

    x: (B, N, d_model) -> out: (B, N, d_head)
    """

    def __init__(self, d_model: int, d_head: int, bias: bool = False) -> None:
        super().__init__()
        self.w_q = nn.Linear(d_model, d_head, bias=bias)
        self.w_k = nn.Linear(d_model, d_head, bias=bias)
        self.w_v = nn.Linear(d_model, d_head, bias=bias)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        q, k, v = self.w_q(x), self.w_k(x), self.w_v(x)  # each (B, N, d_head)
        out, _ = scaled_dot_product_attention(q, k, v, mask=mask)
        return out
