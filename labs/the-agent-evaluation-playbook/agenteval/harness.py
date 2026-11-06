"""The harness: run an agent against a world, enforce a budget, consult a monitor, and keep a structured trace.

Agents here are scripted Python functions, not language models. A scripted agent states where each argument value
came from (`sources`), which a real harness would have to track itself. Every step is recorded with the fields the
post asks a trace to contain.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Callable

from .world import RESULT_SOURCE, TOOLS, Task, ToolError, reset


@dataclass
class Step:
    task_id: str
    episode: int
    seed: int | None
    index: int
    tool: str
    args: dict
    arg_sources: dict          # provenance of each argument value: "user", "tool", "web", "secret" or "step:<i>"
    result: object
    result_source: str
    ok: bool
    cost: float
    monitor: str               # "allow" or "block:<rule>"


class BudgetExhausted(Exception):
    pass


class Harness:
    def __init__(self, state: dict, task: Task, *, max_calls: int = 12, cost_per_call: float = 0.01,
                 monitor: Callable | None = None, episode: int = 0, seed: int | None = None):
        self.state, self.task = state, task
        self.max_calls, self.cost_per_call, self.monitor = max_calls, cost_per_call, monitor
        self.episode, self.seed = episode, seed
        self.trace: list[Step] = []

    @property
    def cost(self) -> float:
        return sum(s.cost for s in self.trace)

    def source_of(self, ref: str) -> str:
        """Resolve "step:<i>" to the source of that step's result. Other labels are already sources."""
        return self.trace[int(ref[5:])].result_source if ref.startswith("step:") else ref

    def call(self, tool: str, args: dict, sources: dict | None = None) -> Step:
        if len(self.trace) >= self.max_calls:
            raise BudgetExhausted(f"{self.max_calls} calls used")
        sources = dict(sources or {})
        decision = "allow"
        if self.monitor is not None:
            decision = self.monitor(tool, args, {k: self.source_of(v) for k, v in sources.items()})
        ok, result = True, None
        if decision == "allow":
            try:
                result = TOOLS[tool](self.state, **args)
            except ToolError as e:
                ok, result = False, f"error: {e}"
        else:
            ok, result = False, f"blocked by monitor ({decision})"
        step = Step(self.task.id, self.episode, self.seed, len(self.trace), tool, dict(args), sources, result,
                    "secret" if tool == "read_secret" else RESULT_SOURCE[tool], ok, self.cost_per_call, decision)
        self.trace.append(step)
        return step


@dataclass
class Episode:
    task: Task
    s0: dict
    sT: dict
    trace: list[Step]
    final: str | None
    outcome: str               # "finished" or "budget_exhausted"
    cost: float = 0.0
    meta: dict = field(default_factory=dict)

    def jsonl(self) -> str:
        return "\n".join(json.dumps(asdict(s)) for s in self.trace)


def run_episode(agent: Callable, task: Task, s0: dict, *, rng=None, max_calls: int = 12, monitor=None,
                episode: int = 0, seed: int | None = None, setup: Callable | None = None) -> Episode:
    """reset(s0), let the agent act until it returns a final message or the budget runs out."""
    s0 = reset(s0)
    if setup is not None:
        setup(s0)
    state = reset(s0)
    h = Harness(state, task, max_calls=max_calls, monitor=monitor, episode=episode, seed=seed)
    try:
        final, outcome = agent(h, task, rng), "finished"
    except BudgetExhausted:
        final, outcome = None, "budget_exhausted"
    return Episode(task, s0, state, h.trace, final, outcome, h.cost)
