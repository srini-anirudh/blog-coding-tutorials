"""Task mixtures: how often each task is drawn, and a sampler that records what it drew.

Four strategies over corpus sizes n_i:

    proportional   p_i = n_i / sum_j n_j               big corpora dominate
    uniform        p_i = 1 / K                         every task equal, small corpora repeat a lot
    temperature    p_i = n_i^(1/T) / sum_j n_j^(1/T)   T = 1 is proportional, T -> inf is uniform
    manual         p_i = w_i / sum_j w_j               hand-tuned weights

Text replay is one more "task" whose share is fixed (for example 10%) before the rest is split.
"""

from __future__ import annotations

import random
from collections import Counter


def mixture_weights(sizes: dict[str, int], strategy: str = "temperature", temperature: float = 2.0,
                    manual: dict[str, float] | None = None) -> dict[str, float]:
    """Sampling probabilities per task. They always sum to 1."""
    if strategy == "proportional":
        raw = {k: float(n) for k, n in sizes.items()}
    elif strategy == "uniform":
        raw = {k: 1.0 for k in sizes}
    elif strategy == "temperature":
        raw = {k: float(n) ** (1.0 / temperature) for k, n in sizes.items()}
    elif strategy == "manual":
        if manual is None or set(manual) != set(sizes):
            raise ValueError("manual mixing needs one weight per task")
        raw = {k: float(manual[k]) for k in sizes}
    else:
        raise ValueError(f"unknown mixture strategy {strategy!r}")
    total = sum(raw.values())
    return {k: v / total for k, v in raw.items()}


def with_replay(weights: dict[str, float], replay_ratio: float, name: str = "text") -> dict[str, float]:
    """Reserve `replay_ratio` of all samples for text-only replay and rescale the rest."""
    if replay_ratio <= 0:
        return dict(weights)
    out = {k: v * (1 - replay_ratio) for k, v in weights.items()}
    out[name] = replay_ratio
    return out


def epochs_per_task(weights: dict[str, float], sizes: dict[str, int], total_samples: int) -> dict[str, float]:
    """How many passes over each corpus a run makes. Above about 4 epochs a small task starts to be memorised."""
    return {k: weights[k] * total_samples / sizes[k] for k in sizes if k in weights}


class MixtureSampler:
    """Draws (task, index) pairs. Each task walks its own shuffled order, so repeats only start after an epoch."""

    def __init__(self, weights: dict[str, float], sizes: dict[str, int], seed: int = 0):
        self.tasks = list(weights)
        self.probs = [weights[t] for t in self.tasks]
        self.sizes = sizes
        self.rng = random.Random(seed)
        self.cursor = Counter()
        self.counts: Counter[str] = Counter()

    def draw(self) -> tuple[str, int]:
        task = self.rng.choices(self.tasks, weights=self.probs, k=1)[0]
        size = self.sizes.get(task, 1 << 30)
        i = self.cursor[task]
        self.cursor[task] += 1
        self.counts[task] += 1
        # A cheap fixed permutation of 0..size-1: an affine map with a multiplier coprime to size.
        return task, (i * _coprime(size) + 7) % size

    def state_dict(self) -> dict:
        return {"rng": self.rng.getstate(), "cursor": dict(self.cursor), "counts": dict(self.counts)}

    def load_state_dict(self, state: dict) -> None:
        self.rng.setstate(_as_state(state["rng"]))
        self.cursor = Counter(state["cursor"])
        self.counts = Counter(state["counts"])

    def observed(self) -> dict[str, float]:
        n = sum(self.counts.values()) or 1
        return {t: self.counts[t] / n for t in self.tasks}


def _coprime(n: int) -> int:
    from math import gcd

    m = 7919  # a prime; fall back to the next number coprime to n
    while gcd(m, n) != 1:
        m += 1
    return m


def _as_state(s):
    """random.Random state survives JSON as nested lists. Turn it back into tuples."""
    return tuple(_as_state(x) for x in s) if isinstance(s, list) else s
