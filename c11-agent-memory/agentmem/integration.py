"""Plug AgentMemory into the C10 harness without editing its loop.

The harness has two extension points, and we use exactly those:

    EventLog.subscribe(fn)      every event reaches the memory hook -> WRITE
    Agent(context=builder)      every model call is built by our builder -> READ

`attach_memory` wires them up. The agent loop itself does not change, which is the
test of whether the harness was designed well.

Every `harness` import below is a top-level name, and C10's own files are never edited.
tests/test_integration.py pins the names and attributes this module relies on, so a
change on the C10 side fails there with a clear message.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

_C10 = Path(__file__).resolve().parents[2] / "c10-agent-harness"
if _C10.exists() and str(_C10) not in sys.path:
    sys.path.insert(0, str(_C10))

from harness import Agent, AgentState, ContextBuilder, Event, Message, ModelResponse  # noqa: E402

from .budget import Block, BudgetError, ContextBudgeter, Section  # noqa: E402
from .consolidate import error_signature  # noqa: E402
from .memory import AgentMemory  # noqa: E402
from .records import Episode  # noqa: E402
from .tokens import count_tokens, truncate_to_tokens  # noqa: E402
from .writer import MemoryEvent  # noqa: E402

SECTIONS = [
    Section("instructions", "instructions"),
    Section("extras", "extras"),                  # the harness sections, such as the repo map
    Section("memory", "memory", max_tokens=400),
    Section("task", "task"),
    Section("events", "events", min_tokens=60),
]
_FRAME = 4  # chat framing tokens per message, the estimate the harness budgets with


def _dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def count_message_tokens(messages: list[dict]) -> int:
    """Estimated tokens for a chat: content, tool-call arguments and framing per message."""
    total = 0
    for m in messages:
        total += _FRAME + count_tokens(m.get("content") or "")
        for c in m.get("tool_calls", []):
            total += count_tokens(c["function"]["name"]) + count_tokens(str(c["function"]["arguments"]))
    return total


def task_section(state: AgentState) -> str:
    return f"## Task\n{state.task}\n\n## Plan\n{state.plan.render()}\n\n## Step\n{state.step}"


@dataclass
class PackedContext:
    """What the Agent reads from a builder's result."""

    messages: list[dict]
    tools: list[dict]
    breakdown: dict[str, int]       # section name -> estimated tokens

    @property
    def tokens(self) -> int:
        return sum(self.breakdown.values())


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
        parts += [f"{c.name} {_dumps(c.arguments)}" for c in m.tool_calls]
    return "\n".join(p for p in parts if p)


class BudgetedContextBuilder(ContextBuilder):
    """A ContextBuilder that packs instructions, memories, task state and turns under B.

    The system message holds the stable text. Older turns are dropped or compressed by
    value density. The first user message and the newest turn are always kept: without
    them the model does not know what it was asked or what just happened.
    """

    def __init__(self, budget: int, memory: AgentMemory | None = None, instructions: str | None = None,
                 clock: Callable[[], float] = lambda: 0.0, k: int = 6,
                 query_fn: Callable[[AgentState], str] | None = None, sections: dict | None = None) -> None:
        super().__init__(instructions=instructions) if instructions else super().__init__()
        self.sections = dict(sections or {})
        self.budget = budget
        self.memory = memory
        self.clock = clock
        self.k = k
        self.query_fn = query_fn or (lambda s: f"{s.task} {s.notes.get('last_error', '')}")
        self.budgeter = ContextBudgeter(SECTIONS)

    def build(self, state: AgentState, history: list[Message], tools: list[dict]) -> PackedContext:
        tools_cost = check_tools_fit(tools, self.budget)
        room = self.budget - tools_cost
        blocks = [Block("instructions", self.instructions, required=True, compressible=False, id="instructions"),
                  Block("task", task_section(state), required=True, id="task")]
        names = []
        for name, fn in self.sections.items():
            if text := fn(state):
                names.append(name)
                blocks.append(Block("extras", text, value=0.6, order=len(names), id=f"extra:{name}"))
        if self.memory is not None:
            if self.memory.working is not None:
                blocks.append(Block("task", self.memory.working.render(), required=True, order=1, id="working"))
            for i, note in enumerate(self.memory.session_notes):
                blocks.append(Block("task", f"today only: {note}", value=1.0, order=2 + i, id=f"note{i}"))
            blocks += self.memory.memory_blocks(self.query_fn(state), self.clock(), self.k)
        turns = group_turns(history)
        share = (room - sum(self.budgeter.cost(b) for b in blocks if b.required)) // 2
        for i, turn in enumerate(turns):
            newest, first = i == len(turns) - 1, i == 0
            text = _render_turn(turn)
            if first and not newest:
                # The two required turns share what is left, or a long opening request
                # would take all of it and leave no room for what just happened.
                text = truncate_to_tokens(text, max(self.budgeter.min_compressed, share))
            blocks.append(Block("events", text, value=0.3 + 0.7 * (i + 1) / len(turns), order=i,
                                required=newest or first, id=f"turn{i}", overhead=_FRAME * len(turn) + 4))
        packed = self.budgeter.pack(blocks, room)
        by_id = {b.id: b for b in packed.blocks}

        def section_text(section: str) -> str:
            return "\n".join(b.text for b in sorted((b for b in packed.blocks if b.section == section),
                                                     key=lambda b: b.order))

        parts = {"instructions": self.instructions}
        parts |= {name: by_id[f"extra:{name}"].text for name in names if f"extra:{name}" in by_id}
        if memory_text := section_text("memory"):
            parts["memory"] = f"## Relevant memory\n{memory_text}"
        parts["task"] = section_text("task")  # last in the system message: closest to the history
        system = "\n\n".join(parts.values())
        kept: list[dict] = []
        for i, turn in enumerate(turns):
            b = by_id.get(f"turn{i}")
            if b is not None:
                kept += _turn_messages(turn, b, self.budgeter)
        breakdown = {name: count_tokens(text) for name, text in parts.items()}
        breakdown["tools"] = tools_cost
        breakdown["history"] = count_message_tokens(kept)
        return PackedContext([{"role": "system", "content": system}] + kept, tools, breakdown)


def check_tools_fit(tools: list[dict], budget: int) -> int:
    """The token cost of the tool schemas, which are not negotiable. Too big is a setup error."""
    cost = count_tokens(_dumps(tools))
    if cost >= budget:
        raise BudgetError(f"the tool schemas alone cost ~{cost} tokens, budget is {budget}: "
                          "raise the budget or register fewer tools")
    return cost


def _turn_messages(turn: list[Message], block: Block, budgeter: ContextBudgeter) -> list[dict]:
    """Messages for one kept turn. A compressed turn shortens every text in proportion.

    Tool arguments shrink value by value, so a kept call is still a well-formed call.
    """
    dicts = [m.to_dict() for m in turn]
    room = budgeter.counter(block.text)
    total = count_tokens(_render_turn(turn))
    if room >= total:
        return dicts

    def shrink(text: str, keep: str = "head") -> str:
        return truncate_to_tokens(text, max(8, room * count_tokens(text) // total), keep=keep)

    for d in dicts:
        if d["content"]:
            d["content"] = shrink(d["content"], "both" if d["role"] == "tool" else "head")
        for c in d.get("tool_calls", []):
            c["function"]["arguments"] = {k: shrink(v) if isinstance(v, str) else v
                                          for k, v in c["function"]["arguments"].items()}
    return dicts


class MemoryHook:
    """EventLog subscriber: turns harness events into memory writes.

    UserMessage -> memory.observe (the writer decides what is kept)
    ToolResult  -> memory.record_step, one (s, a, o, r, t) per call
    Final       -> memory.end_task, one retrievable episode per finished task
    """

    def __init__(self, memory: AgentMemory, state: AgentState, task_id: Callable[[], str] | None = None,
                 session: Callable[[], int] = lambda: 1, clock: Callable[[], float] = lambda: 1.0,
                 entity_of: Callable[[str, dict], str] = lambda name, args: str(next(iter(args.values()), "")),
                 reward_of: Callable[[str, dict, str, bool], float] | None = None) -> None:
        self.memory, self.state = memory, state
        # One id per run. Counting stored tasks keeps ids unique across runs and restarts,
        # and consolidation relies on that: it reads one trajectory per id.
        self.task_id = task_id or (lambda: f"task{len(memory.episodes.tasks()) + 1}")
        self.session, self.clock = session, clock
        self.entity_of = entity_of
        # A retry that worked must score above zero, or consolidation never sees a recovery.
        self.reward_of = reward_of or (lambda name, args, content, is_error: -1.0 if is_error else 1.0)
        self.pending: dict[str, tuple[str, dict]] = {}
        self.current: str | None = None  # the open task, None between runs

    def __call__(self, ev: Event) -> None:
        d: dict[str, Any] = ev.data
        if ev.type == "UserMessage" and d.get("kind") != "feedback":
            if ev.step == 0:
                self.current = self.task_id()
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
                                            self.current or self.task_id(), self.session(), entity))
            self.state.notes["last_error"] = error_signature(d["content"]) or ""
        elif ev.type == "Final":
            self.close_task()

    def close_task(self) -> None:
        """Summarise the open task once. Safe to call again."""
        if self.current is not None:
            self.memory.end_task(self.current)
            self.current = None
        self.state.notes.pop("last_error", None)


def attach_memory(agent: Agent, memory: AgentMemory, budget: int, task_id: Callable[[], str] | None = None,
                  session: Callable[[], int] = lambda: 1, clock: Callable[[], float] = lambda: 1.0,
                  **hook_kwargs: Any) -> MemoryHook:
    """Give an existing Agent long-term memory and a budgeted context. Returns the hook.

    The sections of the agent's current builder, such as the repo map, stay in the
    prompt. Calling `memory.end_session()` between sessions is the caller's job.
    """
    check_tools_fit(agent.tools.schemas(), budget)  # fail here, not on the first model call
    agent.builder = BudgetedContextBuilder(budget, memory, agent.builder.instructions, clock,
                                           sections=agent.builder.sections)
    hook = MemoryHook(memory, agent.state, task_id, session, clock, **hook_kwargs)
    agent.log.subscribe(hook)
    run = agent.run

    def run_then_close(task: str):
        # A run that stops at max_steps or on a model error logs no Final event. Those
        # failures are worth remembering too, and only the return from `run` marks them.
        try:
            return run(task)
        finally:
            hook.close_task()

    agent.run = run_then_close
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
        self.prompts.append(count_message_tokens(messages) + count_tokens(_dumps(tools)))
        return self.fn(messages, tools)
