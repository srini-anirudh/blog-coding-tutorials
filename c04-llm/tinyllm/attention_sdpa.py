"""Fast attention: the same maths as attention_naive, handed to PyTorch's fused SDPA kernel."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .attention_naive import causal_mask


def sdpa_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Causal grouped-query attention via F.scaled_dot_product_attention.

    q: (B, H, T_q, d)    k, v: (B, H_kv, T_kv, d)    returns (B, H, T_q, d)
    """
    q_len, kv_len = q.shape[2], k.shape[2]
    if q_len == kv_len:
        # No cache: the plain lower triangle. is_causal lets the kernel skip masked tiles.
        mask, is_causal = None, True
    elif q_len == 1:
        # One new token attends to everything before it: no mask at all.
        mask, is_causal = None, False
    else:
        # Chunked prefill on top of a cache. is_causal=True would align the triangle to the
        # top-left corner, which is wrong here, so pass the shifted mask explicitly.
        mask, is_causal = causal_mask(q_len, kv_len, q.device), False
    # enable_gqa broadcasts the H_kv heads to H inside the kernel, with no repeat_kv copy.
    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=mask, is_causal=is_causal, enable_gqa=n_rep > 1
    )
