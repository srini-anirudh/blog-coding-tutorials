"""Planning versus execution: a plan has a half-life when other writers share the world.

The task: read L files, found by listing a directory, and write a report with all their values. After every tool
call, with probability q another writer moves one not-yet-read file to a different directory.

    plan-then-act : list once, plan every read, execute the plan. A read of a moved file fails; the report is
                    written with a gap. Success = no move before the plan finishes = (1 - q)^L exactly.
    react         : observe before every action (list, then read). Pays one extra call per step.
    hybrid        : plan once, execute, and re-list only when an observation contradicts the plan (a read error).
"""

from __future__ import annotations

import random


class SharedFolder:
    def __init__(self, n_files: int, q: float, rng: random.Random):
        self.where = {f"f{i}": "inbox" for i in range(n_files)}
        self.values = {f"f{i}": rng.randint(0, 99) for i in range(n_files)}
        self.read: set[str] = set()
        self.q, self.rng, self.calls = q, rng, 0

    def _other_writer(self) -> None:
        unread = [f for f in self.where if f not in self.read]
        if unread and self.rng.random() < self.q:
            self.where[self.rng.choice(unread)] = self.rng.choice(["archive", "shared", "old"])

    def list(self, directory: str) -> list[str]:
        self.calls += 1
        out = sorted(f for f, d in self.where.items() if d == directory)
        self._other_writer()
        return out

    def find(self) -> dict[str, str]:
        """List every directory in one call: where is each file now."""
        self.calls += 1
        out = dict(self.where)
        self._other_writer()
        return out

    def read_file(self, directory: str, name: str) -> int | None:
        self.calls += 1
        ok = self.where.get(name) == directory
        if ok:
            self.read.add(name)
        self._other_writer()
        return self.values[name] if ok else None


def plan_then_act(env: SharedFolder) -> dict:
    plan = [("inbox", f) for f in env.list("inbox")]
    return {f: env.read_file(d, f) for d, f in plan}


def react(env: SharedFolder) -> dict:
    report, todo = {}, env.list("inbox")
    for f in todo:
        while f not in report or report[f] is None:
            loc = env.find()[f]                 # observe first, then act on what was observed
            report[f] = env.read_file(loc, f)
    return report


def hybrid(env: SharedFolder) -> dict:
    plan = {f: "inbox" for f in env.list("inbox")}
    report = {}
    for f in list(plan):
        value = env.read_file(plan[f], f)
        while value is None:                   # contradiction: the belief was wrong, so re-plan
            plan = {**plan, **env.find()}
            value = env.read_file(plan[f], f)
        report[f] = value
    return report


AGENTS = {"plan-then-act": plan_then_act, "react": react, "hybrid": hybrid}


def run(agent: str, n_files: int, q: float, episodes: int, seed: int = 0) -> dict:
    rng = random.Random(seed)
    wins, calls = 0, 0
    for _ in range(episodes):
        env = SharedFolder(n_files, q, rng)
        report = AGENTS[agent](env)
        wins += len(report) == n_files and all(report[f] == env.values[f] for f in env.values)
        calls += env.calls + 1                 # + 1 for writing the report
    return {"success": wins / episodes, "calls": calls / episodes}
