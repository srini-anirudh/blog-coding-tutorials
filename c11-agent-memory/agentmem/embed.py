"""Text embeddings: text -> unit vector, so that cosine similarity is a dot product.

Two embedders share one interface:

- `HashingEmbedder`: signed feature hashing of words and word pairs. No model, no
  download, bit-for-bit deterministic. It matches on shared words, not meaning.
- `TransformerEmbedder`: a small sentence encoder (all-MiniLM-L6-v2 by default) run
  through plain `transformers` with mean pooling. It matches paraphrases.

Shapes: `encode(list of N strings) -> [N, D]` float32, every row L2-normalised.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Protocol

import torch

_WORD_RE = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "a an the is are was were be been of for to in on at by with and or it this that what which "
    "do does did we you i our my me us please can could should would will".split()
)


class Embedder(Protocol):
    dim: int

    def encode(self, texts: list[str]) -> torch.Tensor: ...


def _stem(word: str) -> str:
    """Crude suffix stripping so 'deploys', 'deployed' and 'deploy' share a feature."""
    for suffix in ("ing", "ed", "es", "s"):
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def _features(text: str) -> Counter[str]:
    words = [_stem(w) for w in _WORD_RE.findall(text.lower()) if w not in _STOP]
    feats: Counter[str] = Counter(words)
    feats.update(f"{a}_{b}" for a, b in zip(words, words[1:]))
    return feats


def _bucket(feature: str, dim: int) -> tuple[int, float]:
    # Python's hash() is salted per process; blake2b gives the same bucket on every run.
    h = int.from_bytes(hashlib.blake2b(feature.encode(), digest_size=8).digest(), "little")
    return h % dim, (1.0 if (h >> 63) & 1 else -1.0)


class HashingEmbedder:
    """Signed feature hashing (Weinberger et al., 2009) with sublinear term frequency.

    The random sign makes collisions cancel in expectation instead of always adding,
    so the dot product stays an unbiased estimate of the shared-feature count.
    """

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def encode(self, texts: list[str]) -> torch.Tensor:
        out = torch.zeros(len(texts), self.dim)
        for i, text in enumerate(texts):
            for feat, tf in _features(text).items():
                j, sign = _bucket(feat, self.dim)
                out[i, j] += sign * (1.0 + math.log(tf))
        return torch.nn.functional.normalize(out, dim=-1)  # zero rows stay zero


class TransformerEmbedder:
    """Mean-pooled sentence embeddings from a Hugging Face encoder.

    all-MiniLM-L6-v2 is 6 layers, hidden size 384, about 22.7M parameters, trained
    contrastively on sentence pairs; mean pooling is the pooling it was trained with.
    """

    def __init__(self, name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cpu",
                 batch_size: int = 64) -> None:
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name).to(device).eval()
        self.device = device
        self.batch_size = batch_size
        self.dim = self.model.config.hidden_size

    @torch.no_grad()
    def encode(self, texts: list[str]) -> torch.Tensor:
        chunks = []
        for start in range(0, len(texts), self.batch_size):
            batch = self.tokenizer(texts[start:start + self.batch_size], padding=True, truncation=True,
                                   max_length=256, return_tensors="pt").to(self.device)
            hidden = self.model(**batch).last_hidden_state            # [B, L, D]
            mask = batch["attention_mask"].unsqueeze(-1).float()       # [B, L, 1]
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)  # [B, D]: padding excluded
            chunks.append(torch.nn.functional.normalize(pooled, dim=-1).float().cpu())
        return torch.cat(chunks) if chunks else torch.zeros(0, self.dim)
