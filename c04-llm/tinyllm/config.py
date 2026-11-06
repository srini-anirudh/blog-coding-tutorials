"""Model configuration: every architectural number lives here and nowhere else."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass
class ModelConfig:
    """Hyperparameters of a Llama-style decoder-only transformer.

    The first seven fields are the contract other tutorials rely on (C5 trains this model).
    The rest have defaults that match Llama / SmolLM2 conventions.
    """

    vocab_size: int
    hidden_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    intermediate_size: int
    max_seq_len: int

    head_dim: int | None = None  # None means hidden_size // num_heads
    rope_theta: float = 10_000.0
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = True
    initializer_range: float = 0.02
    attn_backend: str = "sdpa"  # "naive" (reference) or "sdpa" (fused kernel)

    # Mixture of experts. num_experts == 0 means every block uses a dense SwiGLU.
    num_experts: int = 0
    num_experts_per_tok: int = 2
    expert_capacity_factor: float | None = None  # None means no token is ever dropped
    router_aux_loss_coef: float = 0.01

    def __post_init__(self) -> None:
        if self.head_dim is None:
            if self.hidden_size % self.num_heads != 0:
                raise ValueError(f"hidden_size={self.hidden_size} is not divisible by num_heads={self.num_heads}")
            self.head_dim = self.hidden_size // self.num_heads
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(f"num_heads={self.num_heads} must be a multiple of num_kv_heads={self.num_kv_heads}")
        if self.head_dim % 2 != 0:
            raise ValueError("RoPE rotates pairs of channels, so head_dim must be even")
        if self.attn_backend not in ("naive", "sdpa"):
            raise ValueError(f"unknown attn_backend {self.attn_backend!r}")
        if self.num_experts and not 1 <= self.num_experts_per_tok <= self.num_experts:
            raise ValueError("num_experts_per_tok must be between 1 and num_experts")

    @property
    def num_kv_groups(self) -> int:
        """How many query heads share one key/value head (1 = MHA, num_heads = MQA)."""
        return self.num_heads // self.num_kv_heads

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ModelConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "ModelConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))


# Named presets. "tiny" is what the tutorial trains on a laptop. The other two only describe
# real models so the parameter-count code can be checked against published sizes.
PRESETS: dict[str, ModelConfig] = {
    "tiny": ModelConfig(
        vocab_size=256, hidden_size=256, num_layers=6, num_heads=8, num_kv_heads=2,
        intermediate_size=704, max_seq_len=256,
    ),
    "smollm2-135m": ModelConfig(
        vocab_size=49_152, hidden_size=576, num_layers=30, num_heads=9, num_kv_heads=3,
        intermediate_size=1536, max_seq_len=8192, rope_theta=100_000.0,
    ),
    "llama3-8b": ModelConfig(
        vocab_size=128_256, hidden_size=4096, num_layers=32, num_heads=32, num_kv_heads=8,
        intermediate_size=14_336, max_seq_len=8192, rope_theta=500_000.0, tie_word_embeddings=False,
    ),
}


def get_config(name: str, **overrides) -> ModelConfig:
    """Return a fresh copy of a preset, optionally with some fields replaced."""
    base = PRESETS[name].to_dict()
    base["head_dim"] = None  # recompute from hidden_size / num_heads unless overridden
    return ModelConfig.from_dict({**base, **overrides})
