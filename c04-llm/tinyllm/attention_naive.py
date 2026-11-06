"""Reference attention: every step written out, so each intermediate tensor can be printed.

The helpers (split_heads, merge_heads, repeat_kv, causal_mask) are trimmed copies of C3's attnzoo.
"""

from __future__ import annotations

import math

import torch


def split_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    """(B, T, H*d) -> (B, H, T, d). Heads are a reshape of one wide projection, not separate weights."""
    b, t, _ = x.shape
    return x.view(b, t, num_heads, -1).transpose(1, 2)


def merge_heads(x: torch.Tensor) -> torch.Tensor:
    """(B, H, T, d) -> (B, T, H*d). After transpose the memory is out of order, so copy first."""
    b, h, t, d = x.shape
    return x.transpose(1, 2).contiguous().view(b, t, h * d)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """(B, H_kv, T, d) -> (B, H_kv * n_rep, T, d), so each query head finds its shared k/v head.

    Query head h reads kv head h // n_rep. That is the GQA grouping.
    """
    if n_rep == 1:
        return x
    b, h_kv, t, d = x.shape
    return x[:, :, None].expand(b, h_kv, n_rep, t, d).reshape(b, h_kv * n_rep, t, d)


def causal_mask(nq: int, nk: int | None = None, device: torch.device | str | None = None) -> torch.Tensor:
    """Boolean (nq, nk) mask, True where attention is allowed, aligned to the bottom-right corner.

    The queries are the last nq of nk positions, so query i sits at absolute position
    i + (nk - nq) and may see keys up to that position. With nq == nk this is the usual
    lower triangle. With a cache (nq < nk) the diagonal shifts right.
    """
    nk = nq if nk is None else nk
    q_pos = torch.arange(nq, device=device)[:, None] + (nk - nq)
    k_pos = torch.arange(nk, device=device)[None, :]
    return k_pos <= q_pos


def naive_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Causal grouped-query attention.

    q: (B, H, T_q, d)    k, v: (B, H_kv, T_kv, d)    returns (B, H, T_q, d)
    """
    k, v = repeat_kv(k, n_rep), repeat_kv(v, n_rep)
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])  # (B, H, T_q, T_kv)
    mask = causal_mask(q.shape[2], k.shape[2], q.device)
    scores = scores.masked_fill(~mask, float("-inf"))
    # Softmax in float32 so a bf16 run produces the same probabilities as a float32 run.
    probs = torch.softmax(scores.float(), dim=-1).to(q.dtype)
    return probs @ v
