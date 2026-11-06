"""Execution backends behind one function, `attend`.

Architectures (MHA, GQA, MLA, ...) decide *what* q, k, v are. Backends decide
*how* softmax(q k^T / sqrt(d)) v is computed. Keeping them apart means a new
kernel is one entry in BACKENDS, not a change to every attention class.
"""

from __future__ import annotations

from functools import partial
from typing import Callable

import torch
from torch.nn.attention import SDPBackend

from .attention_flash import flash_attention
from .attention_naive import naive_attention
from .attention_sdpa import sdpa_attention
from .attention_tiled import tiled_attention
from .attention_window import chunked_window_attention

BACKENDS: dict[str, Callable[..., torch.Tensor]] = {
    "naive": naive_attention,  # reference: materialises the (Nq, Nk) matrix
    "sdpa": sdpa_attention,  # torch picks the best fused kernel
    "sdpa_math": partial(sdpa_attention, kernel=SDPBackend.MATH),  # torch's unfused C++ path
    "sdpa_flash": partial(sdpa_attention, kernel=SDPBackend.FLASH_ATTENTION),  # CPU flash kernel or CUDA FA2
    "tiled": tiled_attention,  # online softmax in Python, for learning
    "window_chunked": chunked_window_attention,  # sliding window, truly O(N W)
    "flash": flash_attention,  # flash-attn package on CUDA (GPU-only)
}


def attend(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    *,
    backend: str = "sdpa",
    causal: bool = False,
    window_size: int | None = None,
    scale: float | None = None,
    softcap: float | None = None,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """q (B, Hq, Nq, d), k (B, Hkv, Nk, d), v (B, Hkv, Nk, dv) -> (B, Hq, Nq, dv)."""
    if backend not in BACKENDS:
        raise KeyError(f"unknown backend {backend!r}; choose from {sorted(BACKENDS)}")
    if q.size(1) % k.size(1) != 0:
        raise ValueError(f"{q.size(1)} query heads cannot share {k.size(1)} KV heads evenly")
    return BACKENDS[backend](
        q, k, v, attn_mask, causal=causal, window_size=window_size, scale=scale, softcap=softcap, dropout_p=dropout_p
    )
