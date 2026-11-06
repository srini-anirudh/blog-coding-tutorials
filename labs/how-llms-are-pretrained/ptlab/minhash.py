"""MinHash signatures and LSH banding, from scratch.

A document becomes a set of word 5-gram shingles. One MinHash value is the minimum of a random
hash over that set. Two documents get the same minimum with probability equal to their Jaccard
similarity J. With b bands of r rows, two documents become candidates with probability
1 - (1 - J^r)^b.
"""

from __future__ import annotations

import zlib

import numpy as np

MERSENNE = (1 << 31) - 1  # a prime; a * x stays below 2^62, so uint64 never overflows


def shingles(text: str, n: int = 5) -> set[int]:
    """Word n-grams, each hashed to a 31-bit integer. Short documents give one shingle."""
    ws = text.lower().split()
    if len(ws) < n:
        return {zlib.crc32(" ".join(ws).encode()) % MERSENNE}
    return {zlib.crc32(" ".join(ws[i : i + n]).encode()) % MERSENNE for i in range(len(ws) - n + 1)}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a or b else 1.0


class MinHasher:
    """k random hash functions h_i(x) = (a_i * x + b_i) mod p."""

    def __init__(self, k: int = 112, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.k = k
        self.a = rng.integers(1, MERSENNE, size=k, dtype=np.uint64)
        self.b = rng.integers(0, MERSENNE, size=k, dtype=np.uint64)

    def signature(self, items: set[int]) -> np.ndarray:
        x = np.fromiter(items, dtype=np.uint64, count=len(items))
        return ((x[:, None] * self.a[None, :] + self.b[None, :]) % MERSENNE).min(axis=0)


def candidate_prob(j: float | np.ndarray, b: int, r: int):
    """Probability that two documents with Jaccard j share at least one full band."""
    return 1 - (1 - np.asarray(j, dtype=float) ** r) ** b


def threshold(b: int, r: int) -> float:
    """Approximate J where the S-curve rises fastest."""
    return (1 / b) ** (1 / r)


def lsh_candidates(sigs: np.ndarray, b: int, r: int) -> set[tuple[int, int]]:
    """Pairs (i, j), i < j, that agree on every row of at least one band. sigs is (N, b*r)."""
    assert sigs.shape[1] == b * r, "need k = b * r signature values"
    pairs: set[tuple[int, int]] = set()
    for band in range(b):
        buckets: dict[bytes, list[int]] = {}
        for i, row in enumerate(sigs[:, band * r : (band + 1) * r]):
            buckets.setdefault(row.tobytes(), []).append(i)
        for ids in buckets.values():
            for x in range(len(ids)):
                for y in range(x + 1, len(ids)):
                    pairs.add((ids[x], ids[y]))
    return pairs


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self.parent[max(rx, ry)] = min(rx, ry)  # the earlier document becomes the root


def near_dedup(texts: list[str], b: int = 14, r: int = 8, seed: int = 0) -> tuple[list[int], list[list[int]]]:
    """FineWeb-style near-dedup: cluster LSH candidates, keep the first document of each cluster.

    Returns (kept indices, clusters with more than one member).
    """
    mh = MinHasher(b * r, seed)
    sigs = np.stack([mh.signature(shingles(t)) for t in texts]) if texts else np.zeros((0, b * r))
    uf = UnionFind(len(texts))
    for i, j in lsh_candidates(sigs, b, r):
        uf.union(i, j)
    clusters: dict[int, list[int]] = {}
    for i in range(len(texts)):
        clusters.setdefault(uf.find(i), []).append(i)
    kept = sorted(root for root in clusters)
    return kept, [c for c in clusters.values() if len(c) > 1]


def sets_with_jaccard(j: float, size: int, rng: np.random.Generator) -> tuple[set[int], set[int]]:
    """Two random integer sets of equal size whose Jaccard similarity is as close to j as sizes allow.

    With |A| = |B| = size and |A & B| = s, J = s / (2 * size - s), so s = 2 * size * j / (1 + j).
    """
    s = int(round(2 * size * j / (1 + j)))
    pool = rng.choice(MERSENNE, size=2 * size - s, replace=False)
    shared, only_a, only_b = pool[:s], pool[s:size], pool[size:]
    return set(shared.tolist()) | set(only_a.tolist()), set(shared.tolist()) | set(only_b.tolist())
