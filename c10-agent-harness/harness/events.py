"""The event log: every action is one JSON line, appended and never rewritten.

    {"seq": 3, "step": 1, "t": 0.012, "type": "ToolCall", "data": {"id": "call_0001", ...}}

The log is the source of truth. The chat history the model sees is a projection of
it (`to_messages`), which makes three things easy: resuming a crashed run, replaying
a run in a viewer, and auditing what the agent did and who approved it.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from .messages import Message, ToolCall

EVENT_TYPES = ("UserMessage", "ModelResponse", "ToolCall", "ToolResult", "Patch", "TestResult",
               "Approval", "Compaction", "Checkpoint", "Error", "Final")


@dataclass
class Event:
    seq: int
    step: int
    t: float            # seconds since the log was opened
    type: str
    data: dict[str, Any]


class EventLog:
    """Append-only JSONL log. `path=None` keeps it in memory (tests, notebooks)."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self.events: list[Event] = []
        self.subscribers: list[Callable[[Event], None]] = []
        self._t0 = time.monotonic()
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("")  # a fresh run starts a fresh file

    def append(self, type: str, step: int, **data: Any) -> Event:
        if type not in EVENT_TYPES:
            raise ValueError(f"unknown event type {type!r}")
        ev = Event(len(self.events), step, round(time.monotonic() - self._t0, 4), type, data)
        self.events.append(ev)
        if self.path:
            with self.path.open("a") as f:  # one write per event: a crash loses at most one line
                f.write(json.dumps(asdict(ev), ensure_ascii=False) + "\n")
        for fn in self.subscribers:
            fn(ev)
        return ev

    def subscribe(self, fn: Callable[[Event], None]) -> None:
        """Observers (tracers, memory writers, UIs) see every event as it is appended."""
        self.subscribers.append(fn)

    def of_type(self, *types: str) -> list[Event]:
        return [e for e in self.events if e.type in types]

    @classmethod
    def load(cls, path: str | Path) -> "EventLog":
        log = cls()
        for line in Path(path).read_text().splitlines():
            if line.strip():
                log.events.append(Event(**json.loads(line)))
        return log

    def to_messages(self) -> list[Message]:
        """Rebuild the chat history from events, applying compactions in order."""
        history: list[tuple[int, Message]] = []
        for e in self.events:
            d = e.data
            if e.type == "UserMessage":
                history.append((e.seq, Message("user", d["content"])))
            elif e.type == "ModelResponse":
                calls = [ToolCall(c["name"], c["arguments"], id=c["id"]) for c in d.get("tool_calls", [])]
                history.append((e.seq, Message("assistant", d.get("text", ""), calls)))
            elif e.type == "ToolResult":
                history.append((e.seq, Message("tool", d["content"], tool_call_id=d["id"], name=d["name"])))
            elif e.type == "Compaction":
                # Cut by position: an earlier summary has a later seq than the messages it precedes.
                cut = next(i for i, (s, _) in enumerate(history) if s == d["kept_from_seq"])
                history = [(e.seq, Message("user", d["summary"]))] + history[cut:]
        return [m for _, m in history]
