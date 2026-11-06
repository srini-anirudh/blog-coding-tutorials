"""A block-hashed prefix cache, the way inference servers reuse attention state.

A request is a list of token ids. The cache cuts it into blocks of `block_size` tokens
and gives each block a chain hash: the hash of the previous block's hash and this
block's tokens. Two requests share a cached block only if every token before it is
identical too. That is the exact-prefix rule from the post, and it is how vLLM's
automatic prefix caching works. Hosted APIs bill the cached part at a fraction `r`.

The cache stores hashes, not attention state, because here we only count tokens.
`kv_reuse.py` does the same thing with a real model's KV cache.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Request:
    total: int      # tokens sent
    cached: int     # tokens whose state was reused
    computed: int   # tokens the server had to process

    def billed(self, r: float) -> float:
        """Input tokens billed when cached tokens cost a fraction r of the full rate."""
        return self.computed + r * self.cached


def chain_hashes(tokens: list[int], block_size: int) -> list[int]:
    """One hash per full block. Block i's hash depends on blocks 0..i."""
    out, h = [], 0
    for i in range(0, len(tokens) - block_size + 1, block_size):
        h = hash((h, tuple(tokens[i:i + block_size])))
        out.append(h)
    return out


@dataclass
class PrefixCache:
    block_size: int = 16
    blocks: set[int] = field(default_factory=set)

    def lookup(self, tokens: list[int]) -> int:
        """Number of leading tokens already cached (whole blocks only)."""
        n = 0
        for h in chain_hashes(tokens, self.block_size):
            if h not in self.blocks:
                break
            n += self.block_size
        return n

    def insert(self, tokens: list[int]) -> None:
        self.blocks.update(chain_hashes(tokens, self.block_size))

    def request(self, tokens: list[int], use_cache: bool = True) -> Request:
        """Serve one request: reuse the cached prefix, compute the rest, then cache it all."""
        cached = self.lookup(tokens) if use_cache else 0
        self.insert(tokens)
        return Request(len(tokens), cached, len(tokens) - cached)


def first_divergence(a: list[int], b: list[int]) -> int:
    """Index of the first token where two requests differ: everything after it is recomputed."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))
