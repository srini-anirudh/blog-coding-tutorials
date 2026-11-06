"""Chunked sliding-window attention: actually sparse, unlike a window mask.

A window mask on the dense path still computes all Nq x Nk scores, then throws most
of them away. Here we cut the queries into chunks of W. A chunk can only see its
own positions and the W - 1 before them, so one small attention call per chunk is
enough: at most (W, 2W - 1) scores, O(N W) in total. Longformer calls this idea
linear-time local attention; FlashAttention does it with tile skipping.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .attention_tiled import positions_mask


def chunked_window_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    *,
    causal: bool = True,
    window_size: int | None = None,
    scale: float | None = None,
    softcap: float | None = None,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """q (B, Hq, Nq, d), k/v (B, Hkv, Nk, d) -> (B, Hq, Nq, dv). The window is always causal."""
    if window_size is None:
        raise ValueError("backend='window_chunked' needs window_size")
    if attn_mask is not None or softcap is not None:
        raise ValueError("backend='window_chunked' supports neither attn_mask nor softcap")
    nq, nk, w = q.size(-2), k.size(-2), window_size
    offset = nk - nq  # queries are the last nq positions
    scale = 1.0 / math.sqrt(q.size(-1)) if scale is None else scale
    out = q.new_empty(q.size(0), q.size(1), nq, v.size(-1))

    for qs in range(0, nq, w):
        qe = min(qs + w, nq)
        ks = max(0, qs + offset - w + 1)  # the oldest key any query in this chunk can see
        ke = qe + offset  # one past the newest
        mask = positions_mask(qs, qe, ks, ke, offset, True, w, q.device)  # (qe - qs, ke - ks)
        out[:, :, qs:qe] = F.scaled_dot_product_attention(
            q[:, :, qs:qe],
            k[:, :, ks:ke],
            v[:, :, ks:ke],
            attn_mask=mask,
            dropout_p=dropout_p,
            scale=scale,
            enable_gqa=q.size(1) != k.size(1),
        )
    return out
