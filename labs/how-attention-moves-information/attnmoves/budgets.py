"""The arithmetic behind the post's cost claims. Every value here is computed from a formula, not measured.

KV cache:  M_KV = 2 * L * B * S * Hkv * dh * bytes.
Decode:    one new token reads the whole cache once, and does about 4 * S * Hq * dh FLOPs of attention
           per layer (q . k for S keys, then the weighted sum of S values, for each query head).
"""

from __future__ import annotations

GIB = 2**30

# Llama-2-7B shape (the post's "7B-class model"): 32 layers, 32 heads of 128.
LLAMA2_7B = {"layers": 32, "heads": 32, "dh": 128, "params": 6.74e9}
# DeepSeek-V2 attention shape, from its paper: 60 layers, 128 heads of 128, latent 512, RoPE key 64.
DEEPSEEK_V2 = {"layers": 60, "heads": 128, "dh": 128, "d_c": 512, "d_r": 64}


def kv_cache_bytes(layers: int, batch: int, seq: int, kv_heads: int, dh: int, nbytes: int = 2) -> int:
    return 2 * layers * batch * seq * kv_heads * dh * nbytes


def decode_intensity(seq: int, heads: int, kv_heads: int, dh: int, nbytes: int = 2) -> float:
    """Attention FLOPs per byte of cache read, for one decoded token, one layer."""
    flops = 4 * seq * heads * dh
    bytes_read = 2 * seq * kv_heads * dh * nbytes
    return flops / bytes_read


def tp_replication(kv_heads: int, tp: int) -> float:
    """With tensor parallelism over tp ranks, how many copies of each KV head exist."""
    return max(1.0, tp / kv_heads)


def vit_tokens(image: int, patch: int) -> int:
    return (image // patch) ** 2


def receptive_field(window: int, layers: int, exact: bool = False) -> int:
    """How far back a stack of causal sliding windows can reach. The post's estimate is L * w.
    Counting exactly, a window of w keys that includes the query moves at most w - 1 per layer."""
    return layers * (window - 1) if exact else layers * window
