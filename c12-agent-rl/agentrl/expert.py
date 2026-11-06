"""A scripted demonstrator that solves every task from observations alone.

The expert is a generator: it yields an action and receives the tool's answer. ``yield from`` keeps the plan for
each kind of subtask readable. Transient errors are retried, a missing file triggers a search.

Behaviour cloning uses this expert on *clean* environments only. It never shows a recovery, which is exactly the
gap that the distribution-shift section measures.
"""

from __future__ import annotations

import re
from collections.abc import Generator

from .failures import MISSING, TRANSIENT

Plan = Generator[str, str, str]

_SUBTASK = re.compile(
    r"set (?P<key>\S+)(?: in (?P<path>\S+))? to (?:value of (?P<src>\S+)|(?P<value>\S+))$")


def parse_task(text: str) -> list[dict]:
    return [_SUBTASK.match(part).groupdict() for part in text.split(" and ")]


def call(action: str) -> Plan:
    """Issue one tool call; retry while the error is transient. The step budget bounds the retries."""
    obs = yield action
    while obs in TRANSIENT:
        obs = yield action
    return obs


def locate(key: str) -> Plan:
    obs = yield from call(f"search {key}")
    return obs.split()[1]  # "found db/main" -> "db/main"


def lookup_value(key: str) -> Plan:
    path = yield from locate(key)
    obs = yield from call(f"read {path}")
    return dict(kv.split(" = ") for kv in obs.split(" ; "))[key]


def solve(task_text: str) -> Plan:
    for sub in parse_task(task_text):
        value = sub["value"] or (yield from lookup_value(sub["src"]))
        path = sub["path"] or (yield from locate(sub["key"]))
        obs = yield from call(f"edit {path} {sub['key']} {value}")
        if obs == MISSING:  # stale path in the task text: find where the key really lives
            path = yield from locate(sub["key"])
            yield from call(f"edit {path} {sub['key']} {value}")
    yield "submit"
    return "done"


class ScriptedExpert:
    """Callable agent: ``expert(observation) -> action``. It plans from the task text, not from the first
    observation, so it also works after injected mistakes."""

    def __init__(self, task_text: str) -> None:
        self.plan: Plan = solve(task_text)
        self.started = False

    def __call__(self, observation: str) -> str:
        if not self.started:
            self.started = True
            return next(self.plan)
        return self.plan.send(observation)
