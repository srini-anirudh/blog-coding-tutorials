"""The data that flows around the loop: messages, tool calls, model responses.

Everything is a plain dataclass that converts to the OpenAI-style chat dict, because
that is the format both Hugging Face chat templates and OpenAI-compatible servers accept.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]

_ids = itertools.count(1)


def new_call_id() -> str:
    """Short, deterministic ids so event logs diff cleanly between runs."""
    return f"call_{next(_ids):04d}"


def reset_call_ids() -> None:
    """Restart the id counter (tests and demos call this for reproducible logs)."""
    global _ids
    _ids = itertools.count(1)


@dataclass
class ToolCall:
    """One request from the model to run one tool.

    `arguments` is already decoded JSON. `raw` keeps the exact text the model wrote,
    which is what we show back to the model when the call is malformed.
    """

    name: str
    arguments: dict[str, Any]
    id: str = field(default_factory=new_call_id)
    raw: str = field(default="", compare=False)  # provenance, not identity: a replayed call has none

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "type": "function",
                "function": {"name": self.name, "arguments": self.arguments}}


@dataclass
class Message:
    """One chat message. Tool results use role="tool" and point at the call they answer."""

    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            d["tool_calls"] = [c.to_dict() for c in self.tool_calls]
        if self.tool_call_id is not None:
            d["tool_call_id"] = self.tool_call_id
        if self.name is not None:
            d["name"] = self.name
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Message":
        calls = [ToolCall(name=c["function"]["name"], arguments=c["function"]["arguments"], id=c["id"])
                 for c in d.get("tool_calls", [])]
        return cls(role=d["role"], content=d.get("content", ""), tool_calls=calls,
                   tool_call_id=d.get("tool_call_id"), name=d.get("name"))


@dataclass
class Usage:
    """Token accounting for one model call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class ModelResponse:
    """What a model client returns for one call.

    `parse_errors` is non-empty when the model tried to call a tool but wrote
    something we could not decode. The loop feeds those errors back instead of
    treating the turn as a final answer.
    """

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    parse_errors: list[str] = field(default_factory=list)
    raw: str = ""

    @property
    def final(self) -> bool:
        return not self.tool_calls and not self.parse_errors

    def to_message(self) -> Message:
        return Message(role="assistant", content=self.text, tool_calls=list(self.tool_calls))


def dumps(obj: Any) -> str:
    """Compact, key-sorted JSON: stable bytes for logs and for token counts."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
