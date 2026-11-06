"""Reshaping helpers for multi-head attention.

Heads are not separate weights. They are a reshape of one wide projection:
d_model = num_heads * head_dim.
"""

from __future__ import annotations

import torch


def split_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    """(B, N, H*d) -> (B, H, N, d).

    view() splits the last dim into (H, d); transpose() moves H next to B so that
    the matmuls in attention batch over (B, H) and contract over d.
    """
    b, n, _ = x.shape
    return x.view(b, n, num_heads, -1).transpose(1, 2)


def merge_heads(x: torch.Tensor) -> torch.Tensor:
    """(B, H, N, d) -> (B, N, H*d).

    After transpose() the memory is no longer laid out row by row, so view()
    would fail. contiguous() copies it into the order view() needs.
    """
    b, h, n, d = x.shape
    return x.transpose(1, 2).contiguous().view(b, n, h * d)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """(B, Hkv, N, d) -> (B, Hkv*n_rep, N, d): give every query head its KV head.

    Query head h reads KV head h // n_rep. expand() is free (a stride-0 view);
    the reshape afterwards is where the copy happens.
    """
    if n_rep == 1:
        return x
    b, hkv, n, d = x.shape
    return x[:, :, None, :, :].expand(b, hkv, n_rep, n, d).reshape(b, hkv * n_rep, n, d)
