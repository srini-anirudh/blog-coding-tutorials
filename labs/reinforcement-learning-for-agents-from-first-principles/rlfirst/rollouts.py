"""The rollout engine: the straggler tax of a synchronous batch, and the staleness bill of an asynchronous one.

Episode durations are log-normal, T = exp(sigma * Z) with Z standard normal, so the median is 1 time unit and sigma
sets the tail. Nothing here is measured on a real cluster; it is the arithmetic of the post's utilisation bound,
checked by simulation.

Synchronous. Launch N rollouts, wait for all, then take one optimiser step. Each round takes max_i T_i, and the
workers were busy for sum_i T_i of the N * max_i T_i available, so

    utilisation = E[T] / E[max_{i <= N} T_i] .

Asynchronous. N workers restart the instant they finish. Finished trajectories go to a queue. The trainer takes an
optimiser step each time B trajectories are waiting (the step itself is treated as instant). A trajectory started
under policy version v and consumed at version v' has age v' - v. A staleness window k drops any trajectory older
than k, so k = 0 keeps only trajectories that started after the last step.
"""

from __future__ import annotations

import heapq

import numpy as np


def durations(sigma: float, size, rng: np.random.Generator) -> np.ndarray:
    return np.exp(sigma * rng.standard_normal(size))


def sync_utilisation(sigma: float, N: int, rounds: int, rng: np.random.Generator) -> float:
    T = durations(sigma, (rounds, N), rng)
    return float(T.mean() / T.max(1).mean())


def async_run(sigma: float, N: int, B: int, k: int | None, n_steps: int, rng: np.random.Generator) -> dict:
    """Event simulation of N always-busy workers. The trainer steps every B finished trajectories and drops any
    trajectory in the batch that is older than k.

    Returns usable trajectories per time unit, the fraction dropped as too stale, and the age distribution of what
    the trainer used.
    """
    version = 0
    events = [(float(d), w, 0) for w, d in enumerate(durations(sigma, N, rng))]   # (finish time, worker, version)
    heapq.heapify(events)
    queue: list[int] = []
    ages: list[int] = []
    dropped = finished = 0
    now = 0.0
    while version < n_steps:
        now, w, v = heapq.heappop(events)
        finished += 1
        heapq.heappush(events, (now + float(durations(sigma, 1, rng)[0]), w, version))  # restart immediately
        queue.append(v)
        if len(queue) >= B:
            batch, queue = queue[:B], queue[B:]
            for v_start in batch:
                age = version - v_start
                if k is not None and age > k:
                    dropped += 1
                else:
                    ages.append(age)
            version += 1
    used = len(ages)
    a = np.array(ages) if ages else np.zeros(1)
    return {"throughput": used / now, "dropped_frac": dropped / max(1, finished),
            "age_mean": float(a.mean()), "age_p95": float(np.percentile(a, 95)), "age_max": int(a.max())}


def sync_throughput(sigma: float, N: int, rounds: int, rng: np.random.Generator) -> float:
    """Trajectories per time unit when every round waits for its slowest rollout."""
    T = durations(sigma, (rounds, N), rng)
    return float(N * rounds / T.max(1).sum())
