"""Tiny configs shared by the tests: small enough to run in milliseconds on a CPU."""

from __future__ import annotations

import torch

from tinyllm import ModelConfig, TinyLLM


def small_config(**overrides) -> ModelConfig:
    base = dict(
        vocab_size=97, hidden_size=64, num_layers=2, num_heads=4, num_kv_heads=2,
        intermediate_size=160, max_seq_len=64,
    )
    return ModelConfig(**{**base, **overrides})


def small_model(seed: int = 0, **overrides) -> TinyLLM:
    torch.manual_seed(seed)
    return TinyLLM(small_config(**overrides)).eval()


def random_ids(batch: int, seq: int, vocab: int = 97, seed: int = 1) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab, (batch, seq), generator=g)
