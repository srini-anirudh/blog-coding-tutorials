"""Turn model text into structured tool calls, or into precise error feedback.

Models trained on the Hermes / Qwen format write calls like:

    <tool_call>
    {"name": "read_file", "arguments": {"path": "calc.py"}}
    </tool_call>

The parser is strict about structure but forgiving about the small slips that
real models make (arguments as a JSON string, "parameters" instead of "arguments").
Anything it cannot repair becomes a parse error that is shown back to the model.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .messages import ModelResponse, ToolCall, Usage

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", re.S)
FENCED_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)


def _decode_call(blob: str) -> tuple[ToolCall | None, str | None]:
    """Decode one JSON blob into a ToolCall, or explain why not."""
    try:
        obj: Any = json.loads(blob)
    except json.JSONDecodeError as e:
        snippet = blob[max(0, e.pos - 30): e.pos + 30].replace("\n", "\\n")
        return None, f"invalid JSON in tool call ({e.msg} at char {e.pos}): ...{snippet}..."
    if not isinstance(obj, dict):
        return None, f"tool call must be a JSON object, got {type(obj).__name__}"
    name = obj.get("name")
    if not isinstance(name, str) or not name:
        return None, 'tool call needs a string "name" field'
    args = obj.get("arguments", obj.get("parameters", {}))
    if isinstance(args, str):           # some models double-encode the arguments
        try:
            args = json.loads(args) if args.strip() else {}
        except json.JSONDecodeError as e:
            return None, f"arguments for {name!r} are a string that is not valid JSON ({e.msg})"
    if not isinstance(args, dict):
        return None, f"arguments for {name!r} must be a JSON object"
    return ToolCall(name=name, arguments=args, raw=blob), None


def parse_tool_calls(text: str) -> tuple[str, list[ToolCall], list[str]]:
    """Split model output into (visible text, tool calls, parse errors).

    Returns visible text (str), calls (list[ToolCall], zero or more) and
    errors (list[str], zero or more). A turn can contain both good and bad calls.
    """
    blobs = TOOL_CALL_RE.findall(text)
    visible = TOOL_CALL_RE.sub("", text).strip()
    if not blobs:
        # Fallback: a lone fenced JSON object that looks like a call.
        fenced = [b for b in FENCED_RE.findall(text) if '"name"' in b]
        if fenced:
            blobs, visible = fenced, FENCED_RE.sub("", text).strip()
    calls: list[ToolCall] = []
    errors: list[str] = []
    for blob in blobs:
        call, err = _decode_call(blob)
        if call is not None:
            calls.append(call)
        else:
            errors.append(err or "unparseable tool call")
    if "<tool_call>" in text and not blobs:
        errors.append("found <tool_call> but no JSON body")
    return visible, calls, errors


def response_from_text(text: str, usage: Usage | None = None) -> ModelResponse:
    """Build a ModelResponse from raw generated text (used by text-only model clients)."""
    visible, calls, errors = parse_tool_calls(text)
    return ModelResponse(text=visible, tool_calls=calls, parse_errors=errors,
                         usage=usage or Usage(), raw=text)


def format_call_errors(call: ToolCall | None, errors: list[str]) -> str:
    """The feedback message for a call that cannot run. Specific errors fix themselves faster."""
    head = f"Tool call {call.name!r} was rejected:" if call else "Your tool call could not be parsed:"
    lines = [head] + [f"- {e}" for e in errors]
    lines.append('Reply with <tool_call>{"name": ..., "arguments": {...}}</tool_call> using valid JSON.')
    return "\n".join(lines)
