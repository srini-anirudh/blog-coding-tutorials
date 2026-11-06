"""The model ladder: a family of Llama-style decoders that differ only in width and depth.

The model itself is C4's TinyLLM, imported from ../../c04-llm. This file only picks shapes and counts parameters.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

C4 = Path(__file__).resolve().parents[3] / "c04-llm"

VOCAB = 4096  # C5's TinyStories BPE
SEQ = 256


@dataclass(frozen=True)
class Shape:
    d: int  # hidden size
    layers: int
    vocab: int = VOCAB
    seq: int = SEQ

    @property
    def head_dim(self) -> int:
        return 32 if self.d <= 128 else 64

    @property
    def heads(self) -> int:
        return self.d // self.head_dim

    @property
    def ffn(self) -> int:
        """SwiGLU width: 8d/3 rounded up to a multiple of 32, so the MLP has about 8d^2 weights."""
        return -(-8 * self.d // 3 // 32) * 32

    @property
    def linear_params(self) -> int:
        """Weights inside the blocks that multiply activations: q, k, v, o and the three MLP matrices."""
        return self.layers * (4 * self.d * self.d + 3 * self.d * self.ffn)

    @property
    def non_embedding(self) -> int:
        """Kaplan's N: everything except the token embedding (2 RMSNorm vectors per block, plus the final one)."""
        return self.linear_params + (2 * self.layers + 1) * self.d

    @property
    def total(self) -> int:
        """Chinchilla's N: every parameter. Embedding and LM head are tied, so the V x d matrix counts once."""
        return self.non_embedding + self.vocab * self.d

    @property
    def name(self) -> str:
        return f"d{self.d}-L{self.layers}"


# Width grows by about 1.25x per rung and depth with it, so N_total roughly doubles every rung or two.
LADDER = [
    Shape(64, 2), Shape(96, 3), Shape(128, 4), Shape(160, 5), Shape(192, 6), Shape(256, 6),
    Shape(320, 7), Shape(384, 8), Shape(448, 9), Shape(512, 10),
]


def build(shape: Shape):
    """A TinyLLM with this shape (full multi-head attention, tied embeddings)."""
    if str(C4) not in sys.path:
        sys.path.insert(0, str(C4))
    from tinyllm import ModelConfig, TinyLLM  # C4's public contract

    cfg = ModelConfig(
        vocab_size=shape.vocab, hidden_size=shape.d, num_layers=shape.layers, num_heads=shape.heads,
        num_kv_heads=shape.heads, intermediate_size=shape.ffn, max_seq_len=shape.seq,
        tie_word_embeddings=True, attn_backend="sdpa",
    )
    return TinyLLM(cfg)
