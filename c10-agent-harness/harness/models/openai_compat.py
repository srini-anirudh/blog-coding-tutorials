"""Any OpenAI-compatible chat-completions server: OpenAI, vLLM, llama.cpp, Ollama, LM Studio.

Configured entirely by environment variables, so no key ever lands in code or logs:

    OPENAI_BASE_URL   default https://api.openai.com/v1 (vLLM: http://localhost:8000/v1)
    OPENAI_API_KEY    bearer token (local servers usually accept any string)
    HARNESS_MODEL     model name sent in the request

Uses only the standard library, so the harness has no SDK dependency.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from ..messages import ModelResponse, ToolCall, Usage, new_call_id
from .base import ModelError, TransientModelError


def to_wire(messages: list[dict]) -> list[dict]:
    """Our messages keep tool arguments as dicts; the wire format wants JSON strings."""
    out = []
    for m in messages:
        m = dict(m)
        if m.get("tool_calls"):
            m["tool_calls"] = [{**c, "function": {"name": c["function"]["name"],
                                                  "arguments": json.dumps(c["function"]["arguments"])}}
                               for c in m["tool_calls"]]
        out.append(m)
    return out


class OpenAICompatModel:
    def __init__(self, model: str | None = None, base_url: str | None = None,
                 api_key: str | None = None, timeout_s: float = 120.0, temperature: float = 0.0):
        self.name = model or os.environ.get("HARNESS_MODEL", "gpt-4o-mini")
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY", "")
        self.timeout_s = timeout_s
        self.temperature = temperature

    def request_body(self, messages: list[dict], tools: list[dict]) -> dict:
        body = {"model": self.name, "messages": to_wire(messages), "temperature": self.temperature}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        return body

    def complete(self, messages: list[dict], tools: list[dict]) -> ModelResponse:
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(self.request_body(messages, tools)).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                payload = json.loads(r.read())
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:300]
            if e.code == 429 or e.code >= 500:
                raise TransientModelError(f"HTTP {e.code}: {detail}") from None
            raise ModelError(f"HTTP {e.code}: {detail}") from None
        except (urllib.error.URLError, TimeoutError) as e:
            raise TransientModelError(f"connection failed: {e}") from None
        return self.parse(payload)

    @staticmethod
    def parse(payload: dict) -> ModelResponse:
        msg = payload["choices"][0]["message"]
        calls, errors = [], []
        for c in msg.get("tool_calls") or []:
            fn = c.get("function", {})
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError as e:
                errors.append(f"arguments for {fn.get('name')!r} are not valid JSON ({e.msg})")
                continue
            calls.append(ToolCall(fn.get("name", ""), args, id=c.get("id") or new_call_id(),
                                  raw=fn.get("arguments", "")))
        u = payload.get("usage") or {}
        return ModelResponse(text=msg.get("content") or "", tool_calls=calls, parse_errors=errors,
                             usage=Usage(u.get("prompt_tokens", 0), u.get("completion_tokens", 0)),
                             raw=json.dumps(msg))
