"""Two caches from "Serving images at scale": an embedding cache keyed on pixels, and a block prefix cache.

Inside the token sequence every image is the same run of placeholder ids. A prefix cache that hashes only token
ids therefore gives two different pictures of the same size the same key. The fix, as in vLLM's design, mixes
the image's content hash into the hash of every block that contains its tokens.
"""

from __future__ import annotations

import copy
import hashlib

import numpy as np


def image_hash(image, settings: str = "") -> str:
    """Content hash of the pixels plus the preprocessing settings (a different resize is a different entry)."""
    arr = np.asarray(image.convert("RGB"))
    return hashlib.sha256(arr.tobytes() + str(arr.shape).encode() + settings.encode()).hexdigest()[:16]


def block_hashes(token_ids: list[int], block: int = 16, image_token: int | None = None,
                 image_key: str | None = None) -> list[str]:
    """Chained hash of each full block of `block` tokens: hash(previous block hash, ids [, image key])."""
    out, prev = [], ""
    for start in range(0, len(token_ids) - len(token_ids) % block, block):
        ids = token_ids[start:start + block]
        extra = image_key if (image_key and image_token in ids) else ""
        prev = hashlib.sha256(f"{prev}|{ids}|{extra}".encode()).hexdigest()[:16]
        out.append(prev)
    return out


def crop_to(kv, length: int) -> None:
    """Keep the first `length` tokens of a transformers cache (its crop takes a negative count to remove)."""
    extra = kv.get_seq_length() - length
    if extra > 0:
        kv.crop(-extra)


class PrefixCache:
    """Stores the KV cache of each prompt; a lookup returns a copy cropped to the longest matching block prefix."""

    def __init__(self, block: int = 16, use_image_hash: bool = True):
        self.block, self.use_image_hash = block, use_image_hash
        self.entries: dict[str, object] = {}     # block hash -> full KV cache it came from
        self.hits = self.misses = 0

    def _hashes(self, token_ids, image_token, image_key):
        return block_hashes(token_ids, self.block, image_token, image_key if self.use_image_hash else None)

    def lookup(self, token_ids, image_token, image_key):
        """Returns (number of cached tokens, cache cropped to that length, or None)."""
        matched, kv = 0, None
        for i, h in enumerate(self._hashes(token_ids, image_token, image_key)):
            if h not in self.entries:
                break
            matched, kv = (i + 1) * self.block, self.entries[h]
        if kv is None:
            self.misses += 1
            return 0, None
        self.hits += 1
        kv = copy.deepcopy(kv)
        crop_to(kv, matched)
        return matched, kv

    def insert(self, token_ids, image_token, image_key, kv) -> None:
        kv = copy.deepcopy(kv)          # later decoding appends to the live cache; the stored copy must not move
        for h in self._hashes(token_ids, image_token, image_key):
            self.entries.setdefault(h, kv)


class EmbeddingCache:
    """Connector outputs keyed by image hash: a hit skips preprocessing-to-features work in the encoder."""

    def __init__(self):
        self.store, self.hits, self.misses = {}, 0, 0

    def get_or_compute(self, key: str, compute):
        if key in self.store:
            self.hits += 1
        else:
            self.misses += 1
            self.store[key] = compute()
        return self.store[key]
