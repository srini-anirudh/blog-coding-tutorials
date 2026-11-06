"""Fixed evaluation suites, greedy decoding, success rate per suite.

    clean      levels 1-3 from a clean start                      (what the demonstrations cover)
    perturbed  levels 1-3 after 1-2 injected harmless mistakes    (distribution shift)
    failures   level 4: timeouts, tool errors, garbled input, stale paths
    long       level 5: two subtasks, five to seven calls, with failures

Evaluation seeds are 0..n-1; training tasks use seeds >= 10**6, so no evaluation task is ever trained on.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .codec import Codec
from .env import FileSystemEnv
from .rollout import rollout_batch
from .shift import perturbations
from .tasks import sample_task
from .trajectory import Trajectory


@dataclass
class Episode:
    task_level: int
    seed: int
    forced: list[str]

    def env(self, max_steps: int | None = None) -> FileSystemEnv:
        return FileSystemEnv(sample_task(self.task_level, self.seed), max_steps=max_steps)


def suite(name: str, n: int = 150, offset: int = 0) -> list[Episode]:
    eps = []
    for i in range(n):
        seed = offset + i
        if name == "clean":
            eps.append(Episode(1 + i % 3, seed, []))
        elif name == "perturbed":
            level = 1 + i % 3
            eps.append(Episode(level, seed, perturbations(sample_task(level, seed), 1 + i % 2, seed)))
        elif name == "failures":
            eps.append(Episode(4, seed, []))
        elif name == "long":
            eps.append(Episode(5, seed, []))
        else:
            raise ValueError(name)
    return eps


SUITES = ("clean", "perturbed", "failures", "long")


def run_suite(policy, codec: Codec, episodes: list[Episode], *, device: str, temperature: float = 0.0,
              chunk: int = 256) -> list[Trajectory]:
    out = []
    for i in range(0, len(episodes), chunk):
        part = episodes[i: i + chunk]
        out += rollout_batch(policy, codec, [e.env() for e in part], forced=[e.forced for e in part],
                             temperature=temperature, device=device)
    return out


def evaluate(policy, codec: Codec, *, device: str, n: int = 150, suites=SUITES) -> dict[str, float]:
    was_training = policy.training
    policy.eval()
    res = {name: sum(t.success for t in run_suite(policy, codec, suite(name, n), device=device)) / n
           for name in suites}
    policy.train(was_training)
    return res


def pick_device(prefer: str = "auto") -> str:
    if prefer != "auto":
        return prefer
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
