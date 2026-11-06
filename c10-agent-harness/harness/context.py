"""The context builder: what the model sees on each turn, assembled from parts.

    context = instructions + repo state + tools + task state + history

Each part is a named section with its own token count, so the trace can show where
the window goes. Sections are plain callables: adding memory (tutorial C11) means
appending one more section, not editing the loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .messages import Message, dumps
from .planning import Plan
from .tokens import count_message_tokens, count_tokens

DEFAULT_INSTRUCTIONS = """You are a coding agent working inside a repository.
Work in small, verified steps: inspect, reproduce, fix, verify.
Call tools with <tool_call>{"name": ..., "arguments": {...}}</tool_call>.
Read a file before editing it. Prefer edit_file for small changes.
Run the tests after every edit. When the task is done, reply without a tool call
and summarise what you changed."""


@dataclass
class AgentState:
    """Everything the loop knows besides the history. Rendered fresh every turn."""

    task: str
    workspace: Path
    plan: Plan = field(default_factory=Plan)
    step: int = 0
    notes: dict[str, str] = field(default_factory=dict)  # free-form extension slot


Section = Callable[[AgentState], str | None]


def task_section(state: AgentState) -> str:
    return f"## Task\n{state.task}\n\n## Plan\n{state.plan.render()}\n\n## Step\n{state.step}"


@dataclass
class Context:
    messages: list[dict]
    tools: list[dict]
    breakdown: dict[str, int]       # section name -> estimated tokens

    @property
    def tokens(self) -> int:
        return sum(self.breakdown.values())


@dataclass
class ContextBuilder:
    instructions: str = DEFAULT_INSTRUCTIONS
    sections: dict[str, Section] = field(default_factory=dict)   # e.g. {"repo": repo_section}

    def add_section(self, name: str, fn: Section) -> None:
        self.sections[name] = fn

    def build(self, state: AgentState, history: list[Message], tools: list[dict]) -> Context:
        parts: dict[str, str] = {"instructions": self.instructions}
        for name, fn in self.sections.items():
            text = fn(state)
            if text:
                parts[name] = text
        parts["task"] = task_section(state)  # last in the system message: closest to the history
        system = "\n\n".join(parts.values())
        history_dicts = [m.to_dict() for m in history]
        breakdown = {name: count_tokens(text) for name, text in parts.items()}
        breakdown["tools"] = count_tokens(dumps(tools))
        breakdown["history"] = count_message_tokens(history_dicts)
        return Context([{"role": "system", "content": system}] + history_dicts, tools, breakdown)


def repo_section(max_tokens: int = 600) -> Section:
    """git status plus a budgeted repo map, computed once per run and cached."""
    from .git_tools import git, is_repo
    from .repomap import build_repo_map

    cache: dict[str, str] = {}

    def section(state: AgentState) -> str:
        if "map" not in cache:
            cache["map"] = build_repo_map(state.workspace, max_tokens, focus=state.task)
        status = git(state.workspace, "status", "--short").strip() if is_repo(state.workspace) else ""
        return f"## Repository map\n{cache['map']}\n\n## Git status\n{status or 'clean'}"
    return section
