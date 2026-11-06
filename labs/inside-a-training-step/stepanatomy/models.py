"""Small Llama-shaped models, borrowed from C4's TinyLLM rather than rebuilt.

C4 (c04-llm/tinyllm) already builds RMSNorm, RoPE, attention and SwiGLU from scratch. This lab only
needs to run that model and watch it, so it imports it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
if str(REPO / "c04-llm") not in sys.path:
    sys.path.insert(0, str(REPO / "c04-llm"))

from tinyllm import ModelConfig, TinyLLM  # noqa: E402

from .accounting import Shape  # noqa: E402


def config_for(s: Shape, attn: str = "sdpa") -> ModelConfig:
    """A TinyLLM config with the post's conventions: full multi-head attention, untied LM head."""
    return ModelConfig(
        vocab_size=s.vocab, hidden_size=s.hidden, num_layers=s.layers, num_heads=s.heads,
        num_kv_heads=s.heads, intermediate_size=s.intermediate, max_seq_len=s.seq,
        tie_word_embeddings=False, attn_backend=attn,
    )


def build(s: Shape, attn: str = "sdpa", seed: int = 0, device: str = "cpu", dtype=torch.float32) -> TinyLLM:
    torch.manual_seed(seed)
    return TinyLLM(config_for(s, attn)).to(device=device, dtype=dtype)


def random_batch(s: Shape, seed: int = 0, device: str = "cpu") -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, s.vocab, (s.batch, s.seq), generator=g).to(device)


# A CPU-sized shape: the same proportions as Llama-2-7B (I/H = 2.69, head_dim 64), far fewer of everything.
SMALL = Shape(layers=4, hidden=256, heads=4, intermediate=688, vocab=512, seq=256, batch=2)
