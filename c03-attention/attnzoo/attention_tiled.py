"""Tiled attention with an online softmax: the FlashAttention algorithm in plain PyTorch.

This is an educational model of the kernel, not a fast kernel. Python loops over
tiles are slow; the point is that the full (Nq, Nk) score matrix never exists.
At any moment we hold one (block_q, block_k) tile of scores plus three running
statistics per query row:
    m: the largest logit seen so far          (B, H, block_q, 1)
    l: the softmax denominator so far, sum exp(s - m)   (B, H, block_q, 1)
    acc: the unnormalised output so far, sum exp(s - m) v   (B, H, block_q, dv)
When a new tile raises the max from m to m_new, everything accumulated earlier
was computed against the old max, so it is rescaled by exp(m - m_new).
"""

from __future__ import annotations

import math

import torch

from .heads import repeat_kv


def tiled_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    *,
    causal: bool = False,
    window_size: int | None = None,
    scale: float | None = None,
    softcap: float | None = None,
    dropout_p: float = 0.0,
    block_q: int = 64,
    block_k: int = 64,
    trace: list[dict] | None = None,
) -> torch.Tensor:
    """Same contract as naive_attention. If `trace` is a list, one dict per tile is appended."""
    if dropout_p > 0.0:
        raise ValueError("tiled_attention is forward-only and has no dropout")
    n_rep = q.size(1) // k.size(1)
    k, v = repeat_kv(k, n_rep), repeat_kv(v, n_rep)
    b, h, nq, _ = q.shape
    nk = k.size(-2)
    scale = 1.0 / math.sqrt(q.size(-1)) if scale is None else scale
    offset = nk - nq  # queries are the last nq positions (bottom-right alignment)
    if attn_mask is not None:
        attn_mask = attn_mask.expand(b, h, nq, nk)
    out = torch.empty(b, h, nq, v.size(-1), dtype=q.dtype, device=q.device)

    for qs in range(0, nq, block_q):
        qe = min(qs + block_q, nq)
        q_tile = q[:, :, qs:qe].float() * scale
        m = torch.full((b, h, qe - qs, 1), float("-inf"), device=q.device)
        l = torch.zeros(b, h, qe - qs, 1, device=q.device)
        acc = torch.zeros(b, h, qe - qs, v.size(-1), device=q.device)

        for ks in range(0, nk, block_k):
            ke = min(ks + block_k, nk)
            # Skip tiles the mask blocks entirely: this is where causal and local attention save work.
            future = (causal or window_size is not None) and ks > qe - 1 + offset
            too_old = window_size is not None and ke - 1 <= qs + offset - window_size
            if future or too_old:
                if trace is not None:
                    trace.append({"q": (qs, qe), "k": (ks, ke), "skipped": True})
                continue

            s = q_tile @ k[:, :, ks:ke].float().transpose(-2, -1)  # (B, H, bq, bk): one tile of scores
            if softcap is not None:
                s = softcap * torch.tanh(s / softcap)
            tile_mask = positions_mask(qs, qe, ks, ke, offset, causal, window_size, q.device)
            if attn_mask is not None and attn_mask.dtype == torch.bool:
                user = attn_mask[:, :, qs:qe, ks:ke]
                tile_mask = user if tile_mask is None else tile_mask & user
            elif attn_mask is not None:
                s = s + attn_mask[:, :, qs:qe, ks:ke].float()
            if tile_mask is not None:
                s = s.masked_fill(~tile_mask, float("-inf"))

            m_new = torch.maximum(m, s.amax(dim=-1, keepdim=True))
            m_safe = torch.where(torch.isinf(m_new), torch.zeros_like(m_new), m_new)  # rows with nothing visible yet
            p = torch.exp(s - m_safe)  # (B, H, bq, bk), unnormalised probabilities
            alpha = torch.exp(m - m_safe)  # how much the old max overestimated: rescale old sums
            l = alpha * l + p.sum(dim=-1, keepdim=True)
            acc = alpha * acc + p @ v[:, :, ks:ke].float()
            m = m_new
            if trace is not None:
                trace.append({"q": (qs, qe), "k": (ks, ke), "skipped": False, "m": m[0, 0, :, 0].tolist(), "l": l[0, 0, :, 0].tolist()})

        # Divide once at the end. Rows that never saw a key (l = 0) return zeros, like SDPA.
        out[:, :, qs:qe] = torch.where(l > 0, acc / l.clamp_min(1e-30), torch.zeros_like(acc)).to(q.dtype)
    return out


def positions_mask(
    qs: int, qe: int, ks: int, ke: int, offset: int, causal: bool, window_size: int | None, device: torch.device
) -> torch.Tensor | None:
    """The causal/window mask for one tile only, built from absolute positions."""
    if not causal and window_size is None:
        return None
    q_pos = torch.arange(qs, qe, device=device)[:, None] + offset
    k_pos = torch.arange(ks, ke, device=device)[None, :]
    allowed = k_pos <= q_pos
    if window_size is not None:
        allowed = allowed & (k_pos > q_pos - window_size)
    return allowed
