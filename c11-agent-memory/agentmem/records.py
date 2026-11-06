"""The records memory is made of.

`Episode` is one raw transition. `MemoryItem` is anything the retriever can return:
an episode summary, a semantic fact, or a procedure. Both are plain dataclasses so
the whole store serialises to JSON and diffs cleanly.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal

Kind = Literal["episodic", "semantic", "procedural"]
Status = Literal["active", "superseded", "deleted"]
Scope = Literal["global", "project", "session"]

_ids = itertools.count(1)


def new_id(prefix: str) -> str:
    return f"{prefix}{next(_ids):05d}"


def reset_ids() -> None:
    """Restart id counters so runs and tests produce identical stores."""
    global _ids
    _ids = itertools.count(1)


def skip_ids_in(ids: Iterable[str]) -> None:
    """Move the counter past ids read from disk, so a new process cannot reuse one."""
    global _ids
    numbers = [int(m[0]) for i in ids if (m := re.search(r"\d+$", i))]
    start = next(_ids)
    _ids = itertools.count(max([start, *(n + 1 for n in numbers)]))


@dataclass
class Episode:
    """One transition (s, a, o, r, t).

    state:       what the agent was trying to do when it acted (the task state, rendered)
    action:      the tool call it made, as text
    observation: what came back
    reward:      +1 success, -1 failure, 0 neutral
    t:           logical time (session + step / 1000); recency is computed from it
    entity:      the object the action touched; consolidation abstracts over it
    """

    state: str
    action: str
    observation: str
    reward: float
    t: float
    task: str = ""
    session: int = 0
    entity: str = ""
    id: str = field(default_factory=lambda: new_id("ep"))

    def text(self) -> str:
        return f"{self.state} | action: {self.action} | result: {self.observation} | reward: {self.reward:+.0f}"


@dataclass
class MemoryItem:
    """A retrievable memory. `status` is how UPDATE/SUPERSEDE/DELETE are recorded:
    nothing is physically removed, so the history stays auditable."""

    kind: Kind
    text: str
    t: float
    importance: float = 0.5
    confidence: float = 1.0
    scope: Scope = "global"
    source: str = "user"
    status: Status = "active"
    meta: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: new_id("m"))
