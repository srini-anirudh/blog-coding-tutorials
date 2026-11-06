"""Fabric effects that the alpha-beta model leaves out: ECMP hash collisions, packet spraying, MoE imbalance."""

from __future__ import annotations

import math

import numpy as np


def ln_over_lnln(k: int) -> float:
    """The asymptotic max-load estimate for k balls in k bins that the post quotes."""
    return math.log(k) / math.log(math.log(k))


def ecmp_max_load(k_flows: int, k_paths: int, trials: int, rng: np.random.Generator) -> np.ndarray:
    """Hash each flow to a uniformly random uplink. Return the busiest uplink's flow count in each trial."""
    picks = rng.integers(0, k_paths, size=(trials, k_flows))
    counts = np.zeros((trials, k_paths), dtype=np.int64)
    np.add.at(counts, (np.arange(trials)[:, None], picks), 1)
    return counts.max(axis=1)


def spray_max_load(k_flows: int, k_paths: int, packets_per_flow: int, trials: int, rng: np.random.Generator):
    """Spray every packet of every flow to a random uplink (SRD style). Return busiest-link load in units of
    one flow's worth of packets, so 1.0 means perfectly even when k_flows == k_paths."""
    total = k_flows * packets_per_flow
    loads = rng.multinomial(total, [1 / k_paths] * k_paths, size=trials)
    return loads.max(axis=1) / packets_per_flow


def moe_dispatch(
    tokens_per_rank: int,
    ranks: int,
    experts_per_rank: int,
    top_k: int,
    skew: float,
    rng: np.random.Generator,
    capacity_factor: float | None = None,
) -> dict:
    """Route tokens to experts with a skewed router and measure what each rank receives in the AllToAll.

    skew is the standard deviation of per-expert logit offsets: 0 is a balanced router, larger is more popular
    experts. With a capacity factor, each expert keeps at most cf * (average load) tokens and drops the rest.
    """
    e = ranks * experts_per_rank
    bias = rng.normal(0.0, skew, size=e)
    dropped = 0.0
    per_expert = np.zeros(e)
    for _ in range(ranks):
        logits = rng.normal(size=(tokens_per_rank, e)) + bias
        choice = np.argpartition(-logits, top_k, axis=1)[:, :top_k]
        per_expert += np.bincount(choice.ravel(), minlength=e)
    if capacity_factor is not None:
        cap = capacity_factor * per_expert.mean()
        dropped = float(np.clip(per_expert - cap, 0, None).sum())
        per_expert = np.minimum(per_expert, cap)
    recv = per_expert.reshape(ranks, experts_per_rank).sum(axis=1)
    return {
        "max_over_mean": float(recv.max() / recv.mean()),
        "drop_fraction": dropped / (ranks * tokens_per_rank * top_k),
        "recv_tokens": recv,
    }
