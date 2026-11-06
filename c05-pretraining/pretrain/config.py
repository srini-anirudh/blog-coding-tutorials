"""Configuration for a pretraining run.

One dataclass for the model and one for the run. Both round-trip through plain dicts,
so they are saved inside every checkpoint and can be loaded from YAML.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelArgs:
    """Shape of the decoder. The same fields map onto HF LlamaConfig and C4's ModelConfig."""

    vocab_size: int = 4096
    hidden_size: int = 256
    num_layers: int = 6
    num_heads: int = 8
    num_kv_heads: int = 4
    intermediate_size: int = 688
    max_seq_len: int = 256
    rope_theta: float = 10000.0
    tie_embeddings: bool = True


@dataclass
class TrainConfig:
    """Everything that defines a run. Saved in the checkpoint so a resume cannot drift."""

    # data
    data_dir: str = "data_cache/tinystories"
    seq_len: int = 256
    micro_batch_size: int = 16
    grad_accum_steps: int = 2
    num_workers: int = 0
    seed: int = 1337

    # model
    model_type: str = "llama"  # "llama" (HF, standalone) or "tinyllm" (C4)
    model: ModelArgs = field(default_factory=ModelArgs)
    activation_checkpointing: bool = False

    # optimiser
    lr: float = 3e-3
    min_lr: float = 3e-4
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.1
    grad_clip: float = 1.0

    # schedule
    max_steps: int = 2000
    warmup_steps: int = 100

    # precision and device
    precision: str = "bf16"  # "fp32" or "bf16" (autocast)
    device: str = "auto"

    # bookkeeping
    out_dir: str = "runs/tinystories"
    log_every: int = 10
    eval_every: int = 200
    eval_batches: int = 20
    ckpt_every: int = 500
    keep_checkpoints: int = 2
    max_minutes: float = 0.0  # stop cleanly after this many minutes (0 = no limit)

    @property
    def tokens_per_step(self) -> int:
        """Tokens in one optimiser step on one process: B_micro * N_accum * L."""
        return self.micro_batch_size * self.grad_accum_steps * self.seq_len

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TrainConfig":
        d = dict(d)
        model = ModelArgs(**d.pop("model", {}))
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(model=model, **d)

    @classmethod
    def from_yaml(cls, path: str | Path, **overrides: Any) -> "TrainConfig":
        d = yaml.safe_load(Path(path).read_text()) or {}
        model_over = overrides.pop("model", {})
        d.update({k: v for k, v in overrides.items() if v is not None})
        d.setdefault("model", {}).update(model_over)
        return cls.from_dict(d)
