"""Four kinds of state, and the bug you get when the harness forgets that only one of them is in the prompt.

    model context     : ``Harness.context``, a list of lines the policy reads. Compacted when it gets long.
    environment state : ``Calendar.events``. Changed by writes. Not in the model, not in the transcript.
    persistent memory : ``Harness.memory``, a write log the harness may keep outside the context.
    tool state        : ``Calendar.token_calls_left``, an auth token that expires after a few calls.

The policy is a scripted stand-in for a model that trusts its transcript: it works through a to-do list and treats
an item as done only if the context shows a confirmation for it. It is not a language model. It makes the one
assumption the post is about, so the failure is easy to see and to count.

An episode: the user asks for several events, and between them the agent reads long emails (filler that fills the
context). Compaction keeps the task line, a summary of which emails were read, and the most recent lines. The
write confirmations in between are dropped, so the record of each write is gone while the write itself is not.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field


class Calendar:
    """The environment, plus the tool state (an auth token) of the API that changes it."""

    def __init__(self, token_ttl: int = 10**9, idempotent: bool = False):
        self.events: list[str] = []
        self.token_ttl = token_ttl
        self.token_calls_left = token_ttl
        self.idempotent = idempotent
        self.keys: set[str] = set()
        self.calls = 0

    def call(self, tool: str, arg: str = "") -> str:
        self.calls += 1
        if self.token_calls_left <= 0:
            return "error: auth token expired"
        self.token_calls_left -= 1
        if tool == "create_event":
            if self.idempotent and arg in self.keys:      # key = title: the tool refuses a repeat
                return f"ok: {arg} already exists"
            self.keys.add(arg)
            self.events.append(arg)
            return f"ok: created {arg}"
        if tool == "list_events":
            return "events: " + ", ".join(self.events)
        if tool == "read_email":
            return f"email {arg}: " + "lorem " * 40
        raise ValueError(tool)

    def refresh_token(self) -> None:
        self.token_calls_left = self.token_ttl


@dataclass
class HarnessConfig:
    context_budget: int = 12          # lines kept after compaction (plus the task line)
    write_log: bool = False           # persistent memory: re-inject a log of writes after compaction
    check_world: bool = False         # policy lists events before every write
    refresh_token: bool = False       # harness refreshes the token on expiry and retries the call
    idempotent: bool = False          # the tool itself refuses duplicate creates
    token_ttl: int = 10**9


@dataclass
class EpisodeResult:
    duplicates: int
    missing: int
    calls: int
    compactions: int
    errors: int
    finished: bool
    context: list[str] = field(default_factory=list)


def policy(todo: list[str], context: list[str], check_world: bool) -> tuple[str, str] | None:
    """Next action, decided from the context alone. None when every item looks done."""
    for item in todo:
        if item.startswith("read "):
            if not any(line.startswith(f"email {item[5:]}:")
                       or (line.startswith("summary:") and f" {item[5:]} " in line) for line in context):
                return "read_email", item[5:]
            continue
        title = item[7:]
        if any(f"created {title}" in line or f"{title} already exists" in line for line in context):
            continue
        if check_world:                                   # look at the world, not the transcript
            if not context[-1].startswith("events:"):
                return "list_events", ""
            if title in context[-1].split(": ", 1)[1].split(", "):
                continue
        return "create_event", title
    return None


def compact(task: str, context: list[str], keep: int) -> list[str]:
    """A summariser that keeps what looks like content (which emails were read) and drops what looks like
    bookkeeping (tool confirmations such as "ok: created meeting-0")."""
    old, recent = context[1:-keep], context[-keep:]
    read = [line.split()[1].rstrip(":") for line in old if line.startswith("email ")]
    read += [w for line in old if line.startswith("summary: read emails") for w in line.split()[3:]]
    return [task, "summary: read emails " + " ".join(read) + " "] + recent


def run_episode(todo: list[str], cfg: HarnessConfig, max_steps: int = 200) -> EpisodeResult:
    cal = Calendar(cfg.token_ttl, cfg.idempotent)
    task = "task: " + "; ".join(todo)
    context, memory, compactions, errors, finished = [task], [], 0, 0, False
    for _ in range(max_steps):
        action = policy(todo, context, cfg.check_world)
        if action is None:
            finished = True
            break
        obs = cal.call(*action)
        if obs.startswith("error") and cfg.refresh_token:
            cal.refresh_token()                                   # tool state handled by the harness
            obs = cal.call(*action)
        errors += obs.startswith("error")
        if action[0] == "create_event" and obs.startswith("ok"):
            memory.append(obs)
        context.append(obs)
        if len(context) > cfg.context_budget + 1:
            compactions += 1
            context = compact(task, context, keep=cfg.context_budget // 2)
            if cfg.write_log:
                context.insert(1, "write log: " + " | ".join(memory))
    want = [t[7:] for t in todo if t.startswith("create ")]
    dup = sum(cal.events.count(t) - 1 for t in set(cal.events))
    miss = sum(t not in cal.events for t in want)
    return EpisodeResult(dup, miss, cal.calls, compactions, errors, finished, context)


def make_todo(rng: random.Random, n_events: int = 6, reads_between: int = 3) -> list[str]:
    todo = []
    for i in range(n_events):
        todo.append(f"create meeting-{i}")
        todo += [f"read {i}-{j}" for j in range(rng.randint(1, reads_between))]
    return todo
