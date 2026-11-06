"""The unified entry point: AttentionConfig in, attention layer out.

    architecture (attention_type)        execution backend (backend)
    mha | mqa | gqa  -> GroupedQueryAttention     naive | sdpa | sdpa_math
    mla              -> MultiHeadLatentAttention  sdpa_flash | tiled | window_chunked | flash
    linear           -> LinearAttention           (linear attention has its own
    cross            -> CrossAttention             recurrence and ignores backend)
"""

from __future__ import annotations

from torch import nn

from .config import AttentionConfig
from .cross import CrossAttention
from .gqa import GroupedQueryAttention
from .kv_cache import KVCache
from .linear import LinearAttention, LinearStateCache
from .mla import MultiHeadLatentAttention

ARCHITECTURES: dict[str, type[nn.Module]] = {
    "mha": GroupedQueryAttention,
    "mqa": GroupedQueryAttention,
    "gqa": GroupedQueryAttention,
    "mla": MultiHeadLatentAttention,
    "linear": LinearAttention,
    "cross": CrossAttention,
}


def build_attention(config: AttentionConfig, layer_idx: int = 0) -> nn.Module:
    """Instantiate the architecture named by config.attention_type."""
    cls = ARCHITECTURES[config.attention_type]
    return cls(config) if cls is CrossAttention else cls(config, layer_idx=layer_idx)


def make_cache(config: AttentionConfig) -> KVCache | LinearStateCache:
    """The decode-time state each architecture needs. MLA reuses KVCache to hold (c_kv, k_pe)."""
    return LinearStateCache() if config.attention_type == "linear" else KVCache()


def cache_elems_per_token(config: AttentionConfig) -> int:
    """Numbers cached per token per layer. Multiply by L * B * S * bytes for the full cache."""
    t = config.attention_type
    if t == "mla":
        return config.kv_lora_rank + config.qk_rope_head_dim
    if t == "linear":
        return 0  # constant-size state, independent of S: H * dh * (dh + 1) per layer
    return 2 * config.num_kv_heads * config.head_dim
