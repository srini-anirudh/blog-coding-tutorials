"""Action tokenizers: per-step bins (RT-1, RT-2, OpenVLA) and FAST (DCT + rounding + BPE).

Everything here works on numpy arrays. An action chunk has shape (H, D): H time steps, D action dimensions.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict

import numpy as np


# ---------------------------------------------------------------- bins as tokens


class UniformBinner:
    """Map each action dimension to one of `n_bins` uniform bins.

    OpenVLA sets the bin range from the 1st to the 99th percentile of the training actions, so that rare outliers do
    not stretch it. Values outside the range are clipped to the first or last bin.
    """

    def __init__(self, n_bins: int = 256, low_pct: float = 1.0, high_pct: float = 99.0):
        self.n_bins, self.low_pct, self.high_pct = n_bins, low_pct, high_pct
        self.low = self.high = None

    def fit(self, actions: np.ndarray) -> "UniformBinner":
        flat = actions.reshape(-1, actions.shape[-1])
        self.low = np.percentile(flat, self.low_pct, axis=0)
        self.high = np.percentile(flat, self.high_pct, axis=0)
        self.high = np.where(self.high > self.low, self.high, self.low + 1e-6)
        return self

    def normalize(self, a: np.ndarray) -> np.ndarray:
        """Map [low, high] to [-1, 1] per dimension (no clipping)."""
        return 2 * (a - self.low) / (self.high - self.low) - 1

    def encode(self, a: np.ndarray) -> np.ndarray:
        z = np.clip(self.normalize(a), -1, 1)
        return np.minimum(((z + 1) / 2 * self.n_bins).astype(np.int64), self.n_bins - 1)

    def decode(self, bins: np.ndarray) -> np.ndarray:
        """Return the centre of each bin, in the original units."""
        z = (bins + 0.5) / self.n_bins * 2 - 1
        return (z + 1) / 2 * (self.high - self.low) + self.low


def least_used_token_ids(token_counts: dict[int, int], vocab_size: int, k: int = 256,
                         protected: set[int] | None = None) -> list[int]:
    """The `k` vocabulary ids seen least often in a corpus (unseen ids count as zero).

    RT-2 on PaLM-E and OpenVLA on Llama 2 overwrite these ids with the 256 action bins. Ties are broken by the larger
    id, which in BPE vocabularies tends to be a rarer, later merge. `protected` ids (special tokens) are never picked.
    """
    protected = protected or set()
    ids = [i for i in range(vocab_size) if i not in protected]
    ids.sort(key=lambda i: (token_counts.get(i, 0), -i))
    return sorted(ids[:k])


class ActionTokenCodec:
    """Bin an action and write it with reused vocabulary ids, as RT-2 and OpenVLA do."""

    def __init__(self, binner: UniformBinner, action_ids: list[int]):
        assert len(action_ids) == binner.n_bins
        self.binner, self.action_ids = binner, list(action_ids)
        self.id_to_bin = {t: b for b, t in enumerate(self.action_ids)}

    def encode(self, action: np.ndarray) -> list[int]:
        return [self.action_ids[b] for b in self.binner.encode(action).tolist()]

    def decode(self, ids: list[int]) -> np.ndarray:
        return self.binner.decode(np.array([self.id_to_bin[t] for t in ids]))

    def logit_mask(self, vocab_size: int) -> np.ndarray:
        """At inference the output is restricted to the action ids: every other logit gets -inf."""
        mask = np.full(vocab_size, -np.inf)
        mask[self.action_ids] = 0.0
        return mask


def conditional_entropy_bits(tokens: np.ndarray) -> tuple[float, float]:
    """Empirical H(b_t) and H(b_t | b_{t-1}) in bits, for token sequences of shape (N, T).

    This is the "marginal information" per token that FAST argues goes to zero as the control frequency rises.
    """
    marg = Counter(tokens.reshape(-1).tolist())
    total = sum(marg.values())
    h = -sum(c / total * math.log2(c / total) for c in marg.values())
    pairs = Counter(zip(tokens[:, :-1].reshape(-1).tolist(), tokens[:, 1:].reshape(-1).tolist()))
    prev = Counter(tokens[:, :-1].reshape(-1).tolist())
    n = sum(pairs.values())
    hc = -sum(c / n * math.log2(c / prev[a]) for (a, _), c in pairs.items())
    return h, hc


# ---------------------------------------------------------------- FAST


def dct_matrix(n: int) -> np.ndarray:
    """Orthonormal DCT-II matrix M with C = M @ a. The post's formula is the same up to one scale per row."""
    k = np.arange(n)[:, None]
    i = np.arange(n)[None, :]
    m = np.cos(math.pi * k * (2 * i + 1) / (2 * n)) * math.sqrt(2.0 / n)
    m[0] /= math.sqrt(2.0)
    return m


class BytePairEncoder:
    """Byte-pair encoding over sequences of integer symbols, trained from scratch.

    The base vocabulary is every symbol seen in training. Each merge adds one new id for the most frequent adjacent
    pair. The trainer only rescans sequences that contain the pair being merged.
    """

    def __init__(self, vocab_size: int = 1024):
        self.vocab_size = vocab_size
        self.base: dict[int, int] = {}
        self.merges: dict[tuple[int, int], int] = {}
        self.expand: dict[int, tuple[int, ...]] = {}

    def fit(self, seqs: list[list[int]], alphabet: list[int] | None = None) -> "BytePairEncoder":
        """`alphabet` fixes the base symbols, so that values unseen in training can still be encoded."""
        symbols = sorted(set(alphabet) if alphabet is not None else {s for q in seqs for s in q})
        self.base = {s: i for i, s in enumerate(symbols)}
        self.expand = {i: (s,) for s, i in self.base.items()}
        data = [[self.base[s] for s in q] for q in seqs]
        counts: Counter = Counter()
        where: dict[tuple[int, int], set[int]] = defaultdict(set)
        for j, q in enumerate(data):
            for p in zip(q, q[1:]):
                counts[p] += 1
                where[p].add(j)
        next_id = len(self.base)
        while next_id < self.vocab_size and counts:
            pair, c = counts.most_common(1)[0]
            if c < 2:
                break
            self.merges[pair] = next_id
            self.expand[next_id] = self.expand[pair[0]] + self.expand[pair[1]]
            for j in list(where[pair]):
                q = data[j]
                for p in zip(q, q[1:]):
                    counts[p] -= 1
                new = _merge(q, pair, next_id)
                data[j] = new
                for p in zip(new, new[1:]):
                    counts[p] += 1
                    where[p].add(j)
            counts = +counts
            next_id += 1
        return self

    def encode(self, seq: list[int]) -> list[int]:
        q = [self.base[s] for s in seq]
        while len(q) > 1:
            ranked = [(self.merges[p], p) for p in zip(q, q[1:]) if p in self.merges]
            if not ranked:
                break
            new_id, pair = min(ranked)
            q = _merge(q, pair, new_id)
        return q

    def decode(self, ids: list[int]) -> list[int]:
        return [s for i in ids for s in self.expand[i]]


def _merge(q: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    out, i = [], 0
    while i < len(q):
        if i + 1 < len(q) and q[i] == pair[0] and q[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(q[i])
            i += 1
    return out


class FASTTokenizer:
    """The five FAST steps: normalise, DCT per dimension, scale and round, flatten low frequencies first, BPE."""

    def __init__(self, scale: float = 10.0, vocab_size: int = 1024):
        self.scale = scale
        self.binner = UniformBinner()  # only its percentile normalisation is used
        self.bpe = BytePairEncoder(vocab_size)
        self.h = self.d = None

    def coefficients(self, chunk: np.ndarray) -> np.ndarray:
        """Steps 1 to 3: integer DCT coefficients, shape (H, D)."""
        z = np.clip(self.binner.normalize(chunk), -1, 1)
        return np.rint(dct_matrix(chunk.shape[0]) @ z * self.scale).astype(np.int64)

    def flatten(self, coef: np.ndarray) -> list[int]:
        """Step 4: frequency-major order, so every dimension's low-frequency weights come first."""
        return coef.reshape(-1).tolist()

    def fit(self, chunks: np.ndarray) -> "FASTTokenizer":
        self.h, self.d = chunks.shape[1], chunks.shape[2]
        self.binner.fit(chunks)
        # |C_k| <= sqrt(2H) for inputs in [-1, 1], so this alphabet covers every possible rounded coefficient.
        bound = int(math.ceil(math.sqrt(2 * self.h) * self.scale))
        self.bpe.fit([self.flatten(self.coefficients(c)) for c in chunks], alphabet=list(range(-bound, bound + 1)))
        return self

    def encode(self, chunk: np.ndarray) -> list[int]:
        return self.bpe.encode(self.flatten(self.coefficients(chunk)))

    def decode(self, ids: list[int]) -> np.ndarray:
        flat = self.bpe.decode(ids)
        n = self.h * self.d
        flat = (flat + [0] * n)[:n]  # a malformed sequence is padded or cut to the chunk size
        coef = np.array(flat, dtype=np.float64).reshape(self.h, self.d) / self.scale
        z = dct_matrix(self.h).T @ coef
        return (z + 1) / 2 * (self.binner.high - self.binner.low) + self.binner.low


# ---------------------------------------------------------------- synthetic smooth actions


def smooth_actions(n: int, seconds: float, hz: float, dims: int = 7, seed: int = 0,
                   noise: float = 0.0) -> np.ndarray:
    """Smooth end-effector-like action curves: sums of a few slow sinusoids, plus optional white jitter.

    Shape (n, round(seconds * hz), dims). The same seed gives the same underlying curves at every frequency, so a
    sweep over `hz` samples one motion more or less densely. The last dimension is a gripper that switches once.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(int(round(seconds * hz))) / hz
    freqs = rng.uniform(0.2, 1.5, size=(n, 1, dims, 3))
    phase = rng.uniform(0, 2 * math.pi, size=(n, 1, dims, 3))
    amp = rng.uniform(0.2, 1.0, size=(n, 1, dims, 3))
    a = (amp * np.sin(2 * math.pi * freqs * t[None, :, None, None] + phase)).sum(-1)
    switch = rng.uniform(0, seconds, size=(n, 1))
    a[..., -1] = np.where(t[None, :] < switch, 1.0, -1.0)
    if noise:
        a[..., :-1] += noise * rng.standard_normal(a[..., :-1].shape)
    return a
