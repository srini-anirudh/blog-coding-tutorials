"""Plug AgentMemory into the C10 harness without editing its loop.

The harness has two extension points, and we use exactly those:

    EventLog.subscribe(fn)      every event reaches the memory hook -> WRITE
    Agent(context=builder)      every model call is built by our builder -> READ

`attach_memory` wires them up. The agent loop itself does not change, which is the
test of whether the harness was designed well.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Callable

_C10 = Path(__file__).resolve().parents[2] / "c10-agent-harness"
if _C10.exists() and str(_C10) not in sys.path:
    sys.path.insert(0, str(_C10))

from harness import Agent, AgentState, ContextBuilder, Message, ModelResponse  # noqa: E402
from harness.context import Context, task_section  # noqa: E402
from harness.events import Event  # noqa: E402
from harness.messages import dumps  # noqa: E402
from harness.tokens import count_message_tokens  # noqa: E402

from .budget import Block, ContextBudgeter, Section  # noqa: E402
from .consolidate import error_signature  # noqa: E402
from .memory import AgentMemory  # noqa: E402
from .records import Episode  # noqa: E402
from .tokens import count_tokens, truncate_to_tokens  # noqa: E402
from .writer import MemoryEvent  # noqa: E402

SECTIONS = [
    Section("instructions", "instructions"),
    Section("memory", "memory", max_tokens=400),
    Section("task", "task"),
    Section("events", "events", min_tokens=60),
]
_FRAME = 4  # chat framing tokens per message, as in harness.tokens.count_message_tokens


def group_turns(history: list[Message]) -> list[list[Message]]:
    """Group messages so a tool result never loses the assistant call it answers."""
    turns: list[list[Message]] = []
    for m in history:
        if m.role == "tool" and turns:
            turns[-1].append(m)
        else:
            turns.append([m])
    return turns


def _render_turn(turn: list[Message]) -> str:
    parts = []
    for m in turn:
        parts.append(m.content or "")
        parts += [f"{c.name} {dumps(c.arguments)}" for c in m.tool_calls]
    return "\n".join(p for p in parts if p)


class BudgetedContextBuilder(ContextBuilder):
    """A ContextBuilder that packs instructions, memories, task state and turns under B.

    The system message holds the stable text. Older turns are dropped or compressed by
    value density. The first user message and the newest turn are always kept: without
    them the model does not know what it was asked or what just happened.
    """

    def __init__(self, budget: int, memory: AgentMemory | None = None, instructions: str | None = None,
                 clock: Callable[[], float] = lambda: 0.0, k: int = 6,
                 query_fn: Callable[[AgentState], str] | None = None) -> None:
        super().__init__(instructions=instructions) if instructions else super().__init__()
        self.budget = budget
        self.memory = memory
        self.clock = clock
        self.k = k
        self.query_fn = query_fn or (lambda s: f"{s.task} {s.notes.get('last_error', '')}")
        self.budgeter = ContextBudgeter(SECTIONS)

    def build(self, state: AgentState, history: list[Message], tools: list[dict]) -> Context:
        tools_cost = count_tokens(dumps(tools))
        blocks = [Block("instructions", self.instructions, required=True, compressible=False, id="instructions"),
                  Block("task", task_section(state), required=True, id="task")]
        if self.memory is not None:
            blocks += self.memory.memory_blocks(self.query_fn(state), self.clock(), self.k)
        turns = group_turns(history)
        for i, turn in enumerate(turns):
            newest, first = i == len(turns) - 1, i == 0
            blocks.append(Block("events", _render_turn(turn), value=0.3 + 0.7 * (i + 1) / len(turns), order=i,
                                required=newest or first, id=f"turn{i}", overhead=_FRAME * len(turn) + 4))
        packed = self.budgeter.pack(blocks, self.budget - tools_cost)
        by_id = {b.id: b for b in packed.blocks}
        memory_text = "\n".join(b.text for b in sorted((b for b in packed.blocks if b.section == "memory"),
                                                       key=lambda b: b.order))
        parts = {"instructions": self.instructions}
        if memory_text:
            parts["memory"] = f"## Relevant memory\n{memory_text}"
        parts["task"] = by_id["task"].text
        system = "\n\n".join(parts.values())
        kept: list[dict] = []
        for i, turn in enumerate(turns):
            b = by_id.get(f"turn{i}")
            if b is not None:
                kept += _turn_messages(turn, b, self.budgeter)
        breakdown = {name: count_tokens(text) for name, text in parts.items()}
        breakdown["tools"] = tools_cost
        breakdown["history"] = count_message_tokens(kept)
        return Context([{"role": "system", "content": system}] + kept, tools, breakdown)


def _turn_messages(turn: list[Message], block: Block, budgeter: ContextBudgeter) -> list[dict]:
    """Messages for one kept turn. A compressed turn shortens only its tool output."""
    dicts = [m.to_dict() for m in turn]
    room = budgeter.counter(block.text)
    if room >= count_tokens(_render_turn(turn)):
        return dicts
    tool_msgs = [d for d in dicts if d["role"] == "tool"]
    fixed = sum(count_tokens(_render_turn([m])) for m in turn if m.role != "tool")
    each = max(8, (room - fixed) // max(1, len(tool_msgs)))
    for d in tool_msgs:
        d["content"] = truncate_to_tokens(d["content"], each, keep="both")
    return dicts


class MemoryHook:
    """EventLog subscriber: turns harness events into memory writes.

    UserMessage -> memory.observe (the writer decides what is kept)
    ToolResult  -> memory.record_step, one (s, a, o, r, t) per call
    Final       -> memory.end_task, one retrievable episode per finished task
    """

    def __init__(self, memory: AgentMemory, state: AgentState, task_id: Callable[[], str],
                 session: Callable[[], int], clock: Callable[[], float],
                 entity_of: Callable[[str, dict], str] = lambda name, args: str(next(iter(args.values()), "")),
                 reward_of: Callable[[str, dict, str, bool], float] | None = None) -> None:
        self.memory, self.state = memory, state
        self.task_id, self.session, self.clock = task_id, session, clock
        self.entity_of = entity_of
        self.reward_of = reward_of or (lambda name, args, content, is_error: -1.0 if is_error else 0.0)
        self.pending: dict[str, tuple[str, dict]] = {}

    def __call__(self, ev: Event) -> None:
        d: dict[str, Any] = ev.data
        if ev.type == "UserMessage" and d.get("kind") != "feedback":
            # The opening message is a request, not knowledge, unless it states a fact.
            if ev.step > 0 or self.memory.extract(d["content"], self.clock()):
                self.memory.observe(MemoryEvent(d["content"], "user", self.clock()))
        elif ev.type == "ToolCall":
            self.pending[d["id"]] = (d["name"], d["arguments"])
        elif ev.type == "ToolResult":
            name, args = self.pending.pop(d["id"], (d["name"], {}))
            entity = self.entity_of(name, args)
            action = f"{name} {' '.join(str(v) for v in args.values())}".strip()
            reward = self.reward_of(name, args, d["content"], d["is_error"])
            self.memory.record_step(Episode(self.state.task, action, d["content"], reward, self.clock(),
                                            self.task_id(), self.session(), entity))
            self.state.notes["last_error"] = error_signature(d["content"]) or ""
        elif ev.type == "Final":
            self.memory.end_task(self.task_id())
            self.state.notes.pop("last_error", None)


def attach_memory(agent: Agent, memory: AgentMemory, budget: int, task_id: Callable[[], str] = lambda: "task",
                  session: Callable[[], int] = lambda: 1, clock: Callable[[], float] = lambda: 1.0,
                  **hook_kwargs: Any) -> MemoryHook:
    """Give an existing Agent long-term memory and a budgeted context. Returns the hook."""
    agent.builder = BudgetedContextBuilder(budget, memory, agent.builder.instructions, clock)
    hook = MemoryHook(memory, agent.state, task_id, session, clock, **hook_kwargs)
    agent.log.subscribe(hook)
    return hook


class ReaderModel:
    """A harness ModelClient around any `(messages, tools) -> ModelResponse` function.

    It lets a rule policy drive the real loop: same tools, permissions and event log,
    only the thing producing the text is deterministic.
    """

    name = "rule-policy"

    def __init__(self, fn: Callable[[list[dict], list[dict]], ModelResponse]) -> None:
        self.fn = fn
        self.prompts: list[int] = []

    def complete(self, messages: list[dict], tools: list[dict]) -> ModelResponse:
        self.prompts.append(count_message_tokens(messages) + count_tokens(dumps(tools)))
        return self.fn(messages, tools)


_TASK_ID = re.compile(r"\((s\d+t\d+)\)")
