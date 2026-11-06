"""ScriptedModel: a deterministic stand-in for an LLM.

Each step is raw model text (parsed exactly like real model output), a ready-made
ModelResponse, an exception to raise (to test retries), or a function of the
messages so far (to react to tool results). Tests and the demo use it so every
run of the harness is reproducible to the byte.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Union

from ..messages import ModelResponse, Usage
from ..parsing import response_from_text
from ..tokens import count_message_tokens, count_tokens
from .base import ModelError

Step = Union[str, ModelResponse, Exception, Callable[[list[dict]], Union[str, ModelResponse]]]


def call(name: str, **arguments: Any) -> str:
    """Write a tool call the way a Hermes-format model would."""
    return "<tool_call>\n" + json.dumps({"name": name, "arguments": arguments}) + "\n</tool_call>"


class ScriptExhausted(ModelError):
    """The loop asked for more turns than the script has."""


class ScriptedModel:
    name = "scripted"

    def __init__(self, steps: list[Step]):
        self.steps = list(steps)
        self.calls = 0
        self.seen: list[list[dict]] = []   # every context the model was shown, for assertions

    def complete(self, messages: list[dict], tools: list[dict]) -> ModelResponse:
        self.seen.append(messages)
        if self.calls >= len(self.steps):
            raise ScriptExhausted(f"script has {len(self.steps)} steps; turn {self.calls + 1} requested")
        step = self.steps[self.calls]
        self.calls += 1
        if isinstance(step, Exception):
            raise step
        if callable(step) and not isinstance(step, ModelResponse):
            step = step(messages)
        prompt = count_message_tokens(messages) + count_tokens(json.dumps(tools))
        if isinstance(step, ModelResponse):
            step.usage = Usage(prompt, count_tokens(step.raw or step.text))
            return step
        return response_from_text(step, Usage(prompt, count_tokens(step)))
