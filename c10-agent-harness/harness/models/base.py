"""The one interface every model client implements.

    complete(messages, tools) -> ModelResponse

`messages` are OpenAI-style dicts, `tools` are JSON-schema function specs. Whatever
the backend (a script, a local Hugging Face model, a remote API), the loop sees the
same ModelResponse: text, parsed tool calls, parse errors, token usage.
"""

from __future__ import annotations

from typing import Protocol

from ..messages import ModelResponse


class ModelError(Exception):
    """A permanent failure (bad request, auth). The loop stops."""


class TransientModelError(ModelError):
    """A retryable failure (rate limit, timeout, 5xx). The loop backs off and retries."""


class ModelClient(Protocol):
    name: str

    def complete(self, messages: list[dict], tools: list[dict]) -> ModelResponse: ...
