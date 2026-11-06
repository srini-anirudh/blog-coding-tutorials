"""Tokenizers: an existing one (GPT-2) and a small byte-level BPE trained on our corpus.

Both are wrapped in the same tiny interface, so the rest of the pipeline only needs
`encode_batch`, `decode`, `eos_id` and `vocab_size`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

EOS = "<|endoftext|>"


@dataclass
class TextTokenizer:
    """A thin wrapper over a `tokenizers.Tokenizer` with a known EOS token."""

    tok: Tokenizer
    name: str

    @property
    def eos_id(self) -> int:
        return self.tok.token_to_id(EOS)

    @property
    def vocab_size(self) -> int:
        return self.tok.get_vocab_size()

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text).ids

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        # The Rust tokenizer parallelises a batch across threads.
        return [e.ids for e in self.tok.encode_batch(texts)]

    def decode(self, ids: Iterable[int]) -> str:
        return self.tok.decode(list(ids), skip_special_tokens=False)

    def save(self, path: str | Path) -> None:
        self.tok.save(str(path))


def load_pretrained(name: str = "openai-community/gpt2") -> TextTokenizer:
    """An existing tokenizer from the Hub. GPT-2's vocab has 50,257 entries; EOS is id 50256."""
    return TextTokenizer(Tokenizer.from_pretrained(name), name)


def load_file(path: str | Path) -> TextTokenizer:
    return TextTokenizer(Tokenizer.from_file(str(path)), str(path))


def train_bpe(texts: Iterable[str], vocab_size: int = 4096) -> TextTokenizer:
    """Train a byte-level BPE tokenizer.

    Byte-level means every string is encodable (no unknown token): the 256 byte values
    are the base alphabet and BPE merges frequent pairs on top of them.
    EOS is added first so it gets id 0 and is never split.
    """
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[EOS],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(texts, trainer=trainer)
    return TextTokenizer(tok, f"bpe-{vocab_size}")
