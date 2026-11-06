"""AttentionConfig: one description for every attention layer in the zoo.

Architecture fields (attention_type, heads, positions, window, MLA ranks) say
*what* the layer computes. `backend` says *how* the kernel runs. They are
independent: any architecture can run on any backend that supports its features.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

ATTENTION_TYPES = ("mha", "mqa", "gqa", "mla", "linear", "cross")
POSITIONAL = ("rope", "alibi", "none")


@dataclass
class AttentionConfig:
    attention_type: str = "gqa"
    hidden_size: int = 256
    num_heads: int = 8
    num_kv_heads: int | None = None  # None: same as num_heads for mha/gqa, 1 for mqa
    head_dim: int | None = None  # None: hidden_size // num_heads
    causal: bool = True
    window_size: int | None = None  # sliding window: each query sees the last `window_size` keys
    positional_encoding: str = "rope"
    rope_theta: float = 10_000.0
    max_seq_len: int = 4096
    backend: str = "sdpa"  # naive | sdpa | sdpa_math | sdpa_flash | sdpa_efficient | tiled | window_chunked | flash
    qkv_bias: bool = False
    qk_norm: bool = False
    softcap: float | None = None  # logit soft-cap c: s -> c * tanh(s / c)
    dropout: float = 0.0
    # Multi-head latent attention (DeepSeek-V2 naming)
    kv_lora_rank: int = 64
    q_lora_rank: int | None = None
    qk_rope_head_dim: int = 16
    qk_nope_head_dim: int | None = None  # None: head_dim
    v_head_dim: int | None = None  # None: head_dim
    # Cross-attention
    context_size: int | None = None  # width of the sequence we attend to; None: hidden_size
    # Linear attention
    feature_map: str = "elu"

    def __post_init__(self) -> None:
        if self.attention_type not in ATTENTION_TYPES:
            raise ValueError(f"attention_type must be one of {ATTENTION_TYPES}")
        if self.positional_encoding not in POSITIONAL:
            raise ValueError(f"positional_encoding must be one of {POSITIONAL}")
        if self.head_dim is None:
            if self.hidden_size % self.num_heads:
                raise ValueError("hidden_size must be divisible by num_heads (or set head_dim)")
            self.head_dim = self.hidden_size // self.num_heads
        if self.num_kv_heads is None:
            self.num_kv_heads = 1 if self.attention_type == "mqa" else self.num_heads
        if self.attention_type == "mha" and self.num_kv_heads != self.num_heads:
            raise ValueError("mha means num_kv_heads == num_heads")
        if self.attention_type == "mqa" and self.num_kv_heads != 1:
            raise ValueError("mqa means num_kv_heads == 1")
        if self.num_heads % self.num_kv_heads:
            raise ValueError("num_heads must be a multiple of num_kv_heads")
        if self.positional_encoding == "rope" and self.head_dim % 2:
            raise ValueError("RoPE rotates pairs of dimensions, so head_dim must be even")
        self.qk_nope_head_dim = self.qk_nope_head_dim or self.head_dim
        self.v_head_dim = self.v_head_dim or self.head_dim
        self.context_size = self.context_size or self.hidden_size

    @property
    def n_rep(self) -> int:
        """How many query heads share each KV head: Hq / Hkv."""
        return self.num_heads // self.num_kv_heads

    def to_dict(self) -> dict:
        return asdict(self)
