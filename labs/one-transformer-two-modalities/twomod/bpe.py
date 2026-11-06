"""A byte-level BPE tokenizer, trained from scratch.

The text front door is fit once on a corpus and then frozen. It has no gradients: it maps a string to integers.
Training repeatedly merges the most frequent adjacent pair of symbols. Encoding replays the merges in the order
they were learned. Because the base alphabet is the 256 byte values, every string can be encoded and decoding is
exact.
"""

from __future__ import annotations

import re
from collections import Counter

# Split text into words before merging, as GPT-2 does, so merges never cross a word boundary.
# A leading space stays attached to the word: " the" and "the" are different symbols.
PRETOKENIZE = re.compile(r" ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+")


def merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace every occurrence of `pair` in `ids` with `new_id`, left to right."""
    out, i = [], 0
    while i < len(ids):
        if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class ByteBPE:
    def __init__(self, merges: list[tuple[int, int]]) -> None:
        self.merges = list(merges)
        self.ranks = {pair: 256 + i for i, pair in enumerate(self.merges)}  # pair -> id, also its merge order
        self.vocab = {i: bytes([i]) for i in range(256)}
        for (a, b), new_id in self.ranks.items():
            self.vocab[new_id] = self.vocab[a] + self.vocab[b]

    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges)

    @classmethod
    def train(cls, text: str, num_merges: int) -> "ByteBPE":
        word_counts = Counter(PRETOKENIZE.findall(text))
        words = {w: list(w.encode("utf-8")) for w in word_counts}
        merges: list[tuple[int, int]] = []
        for _ in range(num_merges):
            pairs: Counter = Counter()
            for w, ids in words.items():
                for pair in zip(ids, ids[1:]):
                    pairs[pair] += word_counts[w]
            if not pairs:
                break
            best = max(pairs, key=pairs.get)  # ties go to the pair seen first: deterministic
            new_id = 256 + len(merges)
            merges.append(best)
            for w, ids in words.items():
                if len(ids) > 1:
                    words[w] = merge(ids, best, new_id)
        return cls(merges)

    def encode_word(self, ids: list[int]) -> list[int]:
        """Apply the learned merges, earliest first, until none applies."""
        while len(ids) > 1:
            pair = min(zip(ids, ids[1:]), key=lambda p: self.ranks.get(p, float("inf")))
            if pair not in self.ranks:
                break
            ids = merge(ids, pair, self.ranks[pair])
        return ids

    def encode(self, text: str) -> list[int]:
        out: list[int] = []
        for word in PRETOKENIZE.findall(text):
            out += self.encode_word(list(word.encode("utf-8")))
        return out

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def pieces(self, text: str) -> list[str]:
        """The segmentation as readable strings, for printing."""
        return [self.vocab[i].decode("utf-8", errors="replace") for i in self.encode(text)]
