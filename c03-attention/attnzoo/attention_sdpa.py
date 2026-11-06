"""SDPA backend: the same maths, handed to torch.nn.functional.scaled_dot_product_attention.

PyTorch picks a fused kernel (FlashAttention-2, memory-efficient, cuDNN, or a
CPU/MPS kernel) and falls back to the math path when nothing else fits.
"""

from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from .masks import structural_mask


def sdpa_attention(
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
    kernel: SDPBackend | None = None,
) -> torch.Tensor:
    """Shapes as in attention_naive. `kernel` pins one SDPA implementation (e.g. SDPBackend.MATH)."""
    if softcap is not None:
        raise ValueError("SDPA has no logit soft-capping; use backend='naive' (or 'flash' on CUDA).")

    is_causal = False
    nq, nk = q.size(-2), k.size(-2)
    if attn_mask is None and window_size is None and causal and nq == nk:
        is_causal = True  # lets the kernel skip future tiles instead of reading a mask
    else:
        # SDPA's is_causal is aligned top-left, which is wrong when nq < nk (decoding
        # with a cache). Build the bottom-right mask ourselves and merge it in.
        struct = structural_mask(nq, nk, causal, window_size, device=q.device)
        if struct is not None:
            if attn_mask is None:
                attn_mask = struct
            elif attn_mask.dtype == torch.bool:
                attn_mask = attn_mask & struct
            else:
                attn_mask = attn_mask.masked_fill(~struct, float("-inf"))

    ctx = sdpa_kernel([kernel]) if kernel is not None else nullcontext()
    with ctx:
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            enable_gqa=q.size(1) != k.size(1),  # broadcast KV heads inside the kernel, no copy
        )
