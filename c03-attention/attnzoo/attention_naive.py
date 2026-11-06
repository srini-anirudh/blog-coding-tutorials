"""Naive backend: the reference implementation, every intermediate materialised.

All backends share one signature:
    fn(q, k, v, attn_mask=None, *, causal, window_size, scale, softcap, dropout_p) -> out
with q: (B, Hq, Nq, d), k: (B, Hkv, Nk, d), v: (B, Hkv, Nk, dv), out: (B, Hq, Nq, dv).
attn_mask is bool (True = attend) or float (added to the logits), broadcastable
to (B, Hq, Nq, Nk), exactly like torch's scaled_dot_product_attention.
"""

from __future__ import annotations

import math

import torch

from .heads import repeat_kv
from .masks import combine, structural_mask


def naive_attention(
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
    return_weights: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Reference attention. Pass return_weights=True to also get the (B, Hq, Nq, Nk) map."""
    n_rep = q.size(1) // k.size(1)
    k, v = repeat_kv(k, n_rep), repeat_kv(v, n_rep)
    scale = 1.0 / math.sqrt(q.size(-1)) if scale is None else scale

    scores = (q @ k.transpose(-2, -1)) * scale  # (B, H, Nq, Nk)
    if softcap is not None:
        scores = softcap * torch.tanh(scores / softcap)  # squash logits into (-cap, cap)

    allowed = structural_mask(q.size(-2), k.size(-2), causal, window_size, device=q.device)
    if attn_mask is not None and attn_mask.dtype == torch.bool:
        allowed = combine(allowed, attn_mask)
    elif attn_mask is not None:
        scores = scores + attn_mask  # additive biases: ALiBi, relative position, -inf masks
    if allowed is not None:
        scores = scores.masked_fill(~allowed, float("-inf"))

    # Softmax in fp32 even for bf16 inputs: exp() of large logits loses everything in 8 mantissa bits.
    weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    if allowed is not None:
        # A query with no visible key gives softmax(all -inf) = NaN. SDPA returns zeros; match it.
        weights = weights.masked_fill(~allowed.any(dim=-1, keepdim=True), 0.0)
    if dropout_p > 0.0:
        weights = attention_dropout(weights, dropout_p)

    out = weights @ v  # (B, H, Nq, dv)
    return (out, weights) if return_weights else out


def attention_dropout(weights: torch.Tensor, p: float) -> torch.Tensor:
    """Drop whole query-key links, then rescale so the expected output is unchanged.

    Rows no longer sum to 1 in a single sample; they sum to 1 on average.
    """
    keep = torch.rand_like(weights) >= p
    return weights * keep / (1.0 - p)
