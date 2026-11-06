"""Distribution shift: put the agent one small mistake off the expert's path and see what happens.

A perturbed start plays k harmless wrong actions before the policy takes over: a search for an unrelated key, a
read of the wrong file, a malformed call. Each leaves the task fully solvable. The expert recovers from all of them.
A cloned policy has only seen expert prefixes, so after one detour it is in a state no demonstration contains.

To make "a state no demonstration contains" measurable, ``signature`` abstracts a state into the sequence of
(action type, outcome type) pairs that led to it, e.g. ("search:found", "read:ok"). Two states with the same
signature look alike to a policy that has learned the expert's procedure.
"""

from __future__ import annotations

import random

from .tasks import DIRS, KEYS, PATHS, Task
from .trajectory import Trajectory


def random_mistake(task: Task, rng: random.Random) -> str:
    needed = {s.key for s in task.subtasks} | {s.source_key for s in task.subtasks}
    kind = rng.choice(["search", "read", "list", "malformed", "missing"])
    if kind == "search":
        return f"search {rng.choice([k for k in KEYS if k not in needed])}"
    if kind == "read":
        return f"read {rng.choice(sorted(task.files))}"
    if kind == "list":
        return f"list {rng.choice(DIRS)}"
    if kind == "malformed":
        return rng.choice(["edit", "search", "read port", f"edit {rng.choice(sorted(task.files))}"])
    return f"read {rng.choice([p for p in PATHS if p not in task.files])}"


def perturbations(task: Task, k: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    return [random_mistake(task, rng) for _ in range(k)]


def _outcome(obs: str) -> str:
    if obs.startswith("error"):
        return "error"
    return "found" if obs.startswith("found") else "ok"


def signature(traj: Trajectory, t: int) -> tuple[str, ...]:
    """Abstract state before step t: the task level plus the (action type, outcome type) pairs so far."""
    hist = tuple(f"{s.action.split()[0] if s.action.split() else '?'}:{_outcome(s.observation)}"
                 for s in traj.steps[:t])
    return (f"L{traj.level}",) + hist


def seen_signatures(demos: list[Trajectory]) -> set[tuple[str, ...]]:
    return {signature(d, t) for d in demos for t in range(len(d.steps))}


def unseen_rate(trajs: list[Trajectory], seen: set[tuple[str, ...]]) -> float:
    """Fraction of the policy's own decisions (forced steps excluded) taken in a state absent from the demos."""
    states = [signature(tr, t) for tr in trajs for t, s in enumerate(tr.steps) if not s.forced]
    return sum(s not in seen for s in states) / max(len(states), 1)
