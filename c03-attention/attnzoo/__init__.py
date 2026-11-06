"""attnzoo: attention built from scratch, from one head to MLA and tiled online softmax.

Public API (stable; C4 `tinyllm` mirrors GroupedQueryAttention, RotaryEmbedding and KVCache):
"""

from .backends import BACKENDS, attend
from .config import AttentionConfig
from .core import SelfAttentionHead, scaled_dot_product_attention
from .cross import CrossAttention, GatedCrossAttentionBlock
from .gqa import GroupedQueryAttention
from .heads import merge_heads, repeat_kv, split_heads
from .kv_cache import KVCache, StaticKVCache, kv_cache_bytes, mla_cache_bytes
from .library import build_attention, cache_elems_per_token, make_cache
from .linear import LinearAttention, LinearStateCache
from .mha import FusedQKVAttention, MultiHeadAttention, MultiQueryAttention
from .mla import MultiHeadLatentAttention
from .resampler import PerceiverResampler, QFormer
from .rope import RotaryEmbedding, apply_rotary, rotate_half

__all__ = [
    "AttentionConfig",
    "BACKENDS",
    "CrossAttention",
    "FusedQKVAttention",
    "GatedCrossAttentionBlock",
    "GroupedQueryAttention",
    "KVCache",
    "LinearAttention",
    "LinearStateCache",
    "MultiHeadAttention",
    "MultiHeadLatentAttention",
    "MultiQueryAttention",
    "PerceiverResampler",
    "QFormer",
    "RotaryEmbedding",
    "SelfAttentionHead",
    "StaticKVCache",
    "apply_rotary",
    "attend",
    "build_attention",
    "cache_elems_per_token",
    "kv_cache_bytes",
    "make_cache",
    "merge_heads",
    "mla_cache_bytes",
    "repeat_kv",
    "rotate_half",
    "scaled_dot_product_attention",
    "split_heads",
]
