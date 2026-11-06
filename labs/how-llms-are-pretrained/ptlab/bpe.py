"""Byte-level BPE, trained and applied from scratch.

Training starts from the 256 byte values and repeatedly merges the most frequent adjacent pair.
Ties break towards the smaller pair of ids, so training is deterministic. Encoding replays the
merges in the order they were learned. Text is first split with GPT-2's pre-tokenization
pattern, so merges never cross a word boundary.
"""

from __future__ import annotations

from collections import Counter, defaultdict

import regex

GPT2_PATTERN = regex.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

Pair = tuple[int, int]


def pretokenize(text: str) -> list[bytes]:
    return [m.encode("utf-8") for m in GPT2_PATTERN.findall(text)]


def train_bpe(texts: list[str], n_merges: int) -> list[Pair]:
    """Return the merge list. Merge i creates token id 256 + i."""
    word_freq = Counter(w for t in texts for w in pretokenize(t))
    words = [list(w) for w in word_freq]
    freqs = list(word_freq.values())
    pair_count: Counter = Counter()
    where: dict[Pair, set[int]] = defaultdict(set)  # pair -> indices of words that contain it
    for wi, w in enumerate(words):
        for p in zip(w, w[1:]):
            pair_count[p] += freqs[wi]
            where[p].add(wi)
    merges: list[Pair] = []
    for m in range(n_merges):
        if not pair_count:
            break
        best = max(pair_count.items(), key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]))[0]
        new_id = 256 + m
        merges.append(best)
        for wi in list(where[best]):
            w, f = words[wi], freqs[wi]
            for p in zip(w, w[1:]):  # remove this word's old pairs
                pair_count[p] -= f
                if pair_count[p] <= 0:
                    del pair_count[p]
            out, i = [], 0
            while i < len(w):
                if i + 1 < len(w) and (w[i], w[i + 1]) == best:
                    out.append(new_id)
                    i += 2
                else:
                    out.append(w[i])
                    i += 1
            words[wi] = out
            for p in zip(out, out[1:]):  # add the new ones
                pair_count[p] += f
                where[p].add(wi)
        where.pop(best, None)
    return merges


class BPETokenizer:
    def __init__(self, merges: list[Pair]):
        self.merges = merges
        self.rank = {p: i for i, p in enumerate(merges)}
        self.vocab = {i: bytes([i]) for i in range(256)}
        for i, (a, b) in enumerate(merges):
            self.vocab[256 + i] = self.vocab[a] + self.vocab[b]
        self._cache: dict[bytes, list[int]] = {}

    def truncated(self, n_merges: int) -> "BPETokenizer":
        """The same tokenizer with only its first n merges."""
        return BPETokenizer(self.merges[:n_merges])

    def _encode_word(self, w: bytes) -> list[int]:
        if w in self._cache:
            return self._cache[w]
        ids = list(w)
        while len(ids) > 1:
            ranked = [(self.rank.get(p, 1 << 30), i) for i, p in enumerate(zip(ids, ids[1:]))]
            r, i = min(ranked)
            if r == 1 << 30:
                break
            ids[i : i + 2] = [256 + r]
        self._cache[w] = ids
        return ids

    def encode(self, text: str) -> list[int]:
        return [t for w in pretokenize(text) for t in self._encode_word(w)]

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")


def bytes_per_token(n_bytes: int, n_tokens: int) -> float:
    return n_bytes / max(n_tokens, 1)


def training_flops(n_params: float, n_bytes: float, tokens_per_byte: float) -> float:
    """C ~ 6 N D, with D = bytes * tokens per byte (Kaplan et al., 2020)."""
    return 6 * n_params * n_bytes * tokens_per_byte


def logits_bytes(batch: int, seq: int, vocab: int, bytes_per_elem: int = 4) -> int:
    """Size of one fp32 logits tensor, batch x sequence x vocabulary."""
    return batch * seq * vocab * bytes_per_elem
