"""A byte-level tokenizer: every UTF-8 byte is one token, so the vocabulary is exactly 256.

It needs no training and no download, which keeps this tutorial about the model.
C5 replaces it with a trained BPE tokenizer.
"""

from __future__ import annotations


class ByteTokenizer:
    vocab_size = 256

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: list[int]) -> str:
        # errors="replace": a sampled byte sequence is not always valid UTF-8.
        return bytes(ids).decode("utf-8", errors="replace")
