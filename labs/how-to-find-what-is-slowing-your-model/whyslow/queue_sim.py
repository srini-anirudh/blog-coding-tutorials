"""A producer, a bounded queue and an expensive consumer: the input pipeline as a small simulation.

The model follows torch.utils.data: `n_workers` processes each build one batch at a time, at most
`n_workers * prefetch` batches are in flight or waiting, and batches are handed to the consumer in order.
The consumer (the GPU step) takes `gpu_ms` per batch. Everything is deterministic unless `jitter` > 0.
"""

from __future__ import annotations

import random


def simulate(n_workers: int, worker_ms: float, gpu_ms: float, prefetch: int = 2, steps: int = 300,
             prefill: bool = False, jitter: float = 0.0, seed: int = 0) -> dict:
    """Returns per-step wait (time the consumer sat idle before batch j was ready) and the queue depth
    (batches ready and waiting) at the moment the consumer asked for batch j. Depth 0 means it waits."""
    rng = random.Random(seed)
    cap = n_workers * prefetch
    worker_free = [0.0] * n_workers
    done: list[float] = []   # done[i]: when batch i was ready
    take: list[float] = []   # take[j]: when the consumer took batch j
    waits, depth = [], []

    def issue(i: int) -> None:
        if prefill and i < cap:
            done.append(0.0)
            return
        slot_open = take[i - cap] if i >= cap else 0.0  # a slot frees when batch i - cap is taken
        w = min(range(n_workers), key=worker_free.__getitem__)
        start = max(worker_free[w], slot_open)
        cost = worker_ms * (1.0 + jitter * (2 * rng.random() - 1)) if jitter else worker_ms
        worker_free[w] = start + cost
        done.append(start + cost)

    t_free = 0.0
    for j in range(steps):
        while len(done) < min(j + cap, steps):
            issue(len(done))
        depth.append(sum(1 for i in range(j, len(done)) if done[i] <= t_free))
        t = max(t_free, done[j])
        waits.append(t - t_free)
        take.append(t)
        t_free = t + gpu_ms
    return {"wait_ms": waits, "depth": depth, "cap": cap, "total_ms": t_free}


def steady_state(res: dict, skip: int = 100) -> dict:
    w, d = res["wait_ms"][skip:], res["depth"][skip:]
    hist = {k: d.count(k) for k in range(res["cap"] + 1)}
    return {"mean_wait_ms": sum(w) / len(w), "mean_depth": sum(d) / len(d), "depth_hist": hist,
            "at_zero": d.count(0) / len(d), "at_cap": d.count(res["cap"]) / len(d)}


def steps_until_dry(res: dict, tol: float = 1e-9) -> int | None:
    """First step that had to wait. With a prefilled queue this is how long depth hides a deficit."""
    for j, w in enumerate(res["wait_ms"]):
        if w > tol:
            return j
    return None
