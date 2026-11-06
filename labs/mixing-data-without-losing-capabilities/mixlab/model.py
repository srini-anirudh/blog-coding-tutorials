"""The model: the Llama-style TinyLLM built in C4, shrunk until the tasks have to compete for capacity.

The lab does not rebuild a transformer. It imports `tinyllm` from `c04-llm/` (two folders up) and only changes
the configuration: a 21-symbol vocabulary, a small width and two layers.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_C04 = Path(__file__).resolve().parents[3] / "c04-llm"
if str(_C04) not in sys.path:
    sys.path.insert(0, str(_C04))

from tinyllm import ModelConfig, TinyLLM  # noqa: E402

from .tasks import VOCAB  # noqa: E402


def make_model(hidden: int = 48, layers: int = 2, heads: int = 4, seed: int = 0, seq_len: int = 24) -> TinyLLM:
    torch.manual_seed(seed)
    cfg = ModelConfig(
        vocab_size=len(VOCAB),
        hidden_size=hidden,
        num_layers=layers,
        num_heads=heads,
        num_kv_heads=heads,
        intermediate_size=hidden * 2,
        max_seq_len=seq_len,
        tie_word_embeddings=True,
        attn_backend="sdpa",
    )
    return TinyLLM(cfg)


def n_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
