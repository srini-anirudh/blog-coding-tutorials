"""Flash backend: the real FlashAttention kernels. GPU-only (CUDA).

Two routes, tried in order:
1. The `flash-attn` package (Tri Dao's kernels). Supports causal, sliding
   windows, soft-capping and GQA natively. Needs fp16/bf16 on an Ampere+ GPU.
2. PyTorch's built-in FlashAttention-2 kernel via SDPA, pinned with sdpa_kernel.
   No window, soft-cap or dense mask, so those raise. A causal prefill and a
   one-token decode step both work; several queries over a longer cache do not.
Neither runs on CPU or Apple MPS; the tests for this file are marked `gpu`.
PyTorch's separate CPU flash kernel is a different program. It is the `sdpa_flash` backend.
"""

from __future__ import annotations

import torch
from torch.nn.attention import SDPBackend

from .attention_sdpa import sdpa_attention


def flash_available() -> bool:
    """True when a CUDA device exists (the only place these kernels run)."""
    return torch.cuda.is_available()


def flash_attention(
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
) -> torch.Tensor:
    """Shapes as in attention_naive: q (B, Hq, Nq, d), k/v (B, Hkv, Nk, d)."""
    if q.device.type != "cuda":
        raise RuntimeError("backend='flash' is GPU-only; use 'sdpa' or 'tiled' on CPU/MPS")
    if attn_mask is not None:
        # Flash kernels take no dense mask; padding goes through the varlen API (cu_seqlens).
        raise ValueError("flash backend takes no attn_mask; unpad and use flash_attn_varlen_func")
    try:
        from flash_attn import flash_attn_func
    except ImportError:
        if window_size is not None or softcap is not None:
            raise ValueError("window/softcap need the flash-attn package") from None
        return sdpa_attention(q, k, v, causal=causal, scale=scale, dropout_p=dropout_p, kernel=SDPBackend.FLASH_ATTENTION)

    # flash-attn wants (B, N, H, d). Its causal mask is aligned bottom-right, like ours.
    out = flash_attn_func(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        dropout_p=dropout_p,
        softmax_scale=scale,
        causal=causal or window_size is not None,
        window_size=(window_size - 1, 0) if window_size is not None else (-1, -1),
        softcap=softcap or 0.0,
    )
    return out.transpose(1, 2)
