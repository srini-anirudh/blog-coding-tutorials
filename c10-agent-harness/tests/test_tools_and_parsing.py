"""Step 2-3: schema generation from type hints, validation, and parse-error feedback."""

from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from harness.messages import ToolCall
from harness.parsing import format_call_errors, parse_tool_calls, response_from_text
from harness.permissions import Risk
from harness.tools import ToolContext, ToolError, ToolRegistry, build_tool, tool, type_to_schema


class Color(Enum):
    RED = "red"
    BLUE = "blue"


@tool(risk=Risk.READ)
def sample(ctx: ToolContext, path: str, n: int = 3, tags: list[str] | None = None,
           mode: Literal["a", "b"] = "a", ratio: Optional[float] = None) -> str:
    """Do a sample thing.

    Second paragraph is not part of the summary.

    Args:
        path: Where to look.
        n: How many
            lines to show.
    """
    return f"{path}:{n}"


def test_type_to_schema_primitives_and_containers():
    assert type_to_schema(int) == {"type": "integer"}
    assert type_to_schema(list[str]) == {"type": "array", "items": {"type": "string"}}
    assert type_to_schema(Optional[bool]) == {"type": "boolean"}
    assert type_to_schema(Literal["x", "y"]) == {"type": "string", "enum": ["x", "y"]}
    assert type_to_schema(Color) == {"type": "string", "enum": ["red", "blue"]}
    assert type_to_schema(int | str) == {"anyOf": [{"type": "integer"}, {"type": "string"}]}


def test_schema_from_signature_and_docstring():
    fn = sample.__tool__.schema()["function"]
    assert fn["name"] == "sample"
    assert fn["description"] == "Do a sample thing."
    params = fn["parameters"]
    assert params["required"] == ["path"]                  # defaults are optional
    assert "ctx" not in params["properties"]               # the context is injected, never shown
    assert params["properties"]["n"]["default"] == 3
    assert params["properties"]["n"]["description"] == "How many lines to show."
    assert params["properties"]["mode"]["enum"] == ["a", "b"]
    assert params["additionalProperties"] is False


def test_duplicate_tool_name_rejected():
    reg = ToolRegistry([sample])
    try:
        reg.register(sample)
    except ValueError as e:
        assert "duplicate" in str(e)
    else:
        raise AssertionError("duplicate registration should fail")


def test_validate_reports_every_problem():
    reg = ToolRegistry([sample])
    errs = reg.validate(ToolCall("sample", {"n": "many", "bogus": 1, "mode": "z"}))
    joined = "\n".join(errs)
    assert "missing required argument 'path'" in joined
    assert "unexpected argument 'bogus'" in joined
    assert "expected integer" in joined
    assert "is not one of" in joined


def test_validate_unknown_tool_lists_alternatives():
    reg = ToolRegistry([sample])
    assert "available tools: sample" in reg.validate(ToolCall("smaple", {}))[0]


def test_coercion_of_quoted_numbers_can_be_disabled():
    call = ToolCall("sample", {"path": "x", "n": "12"})
    assert ToolRegistry([sample]).validate(call) == [] and call.arguments["n"] == 12
    strict = ToolRegistry([sample], coerce=False)
    assert strict.validate(ToolCall("sample", {"path": "x", "n": "12"}))


def test_bool_is_not_an_integer():
    errs = ToolRegistry([sample]).validate(ToolCall("sample", {"path": "x", "n": True}))
    assert "expected integer" in errs[0]


def test_execute_turns_exceptions_into_error_outputs():
    def boom() -> str:
        """Fail."""
        raise RuntimeError("kaput")

    def expected() -> str:
        """Fail politely."""
        raise ToolError("no such thing")

    reg = ToolRegistry([build_tool(boom), build_tool(expected)])
    assert reg.execute(ToolCall("boom", {})).is_error
    assert "RuntimeError: kaput" in reg.execute(ToolCall("boom", {})).content
    assert reg.execute(ToolCall("expected", {})).content == "Error: no such thing"


def test_parse_hermes_call():
    text, calls, errors = parse_tool_calls('Let me look.\n<tool_call>\n{"name": "read_file", "arguments": {"path": "a.py"}}\n</tool_call>')
    assert text == "Let me look."
    assert calls[0].name == "read_file" and calls[0].arguments == {"path": "a.py"}
    assert errors == []


def test_parse_repairs_small_slips():
    # Double-encoded arguments, and "parameters" instead of "arguments".
    _, calls, errors = parse_tool_calls('<tool_call>{"name": "x", "arguments": "{\\"a\\": 1}"}</tool_call>'
                                        '<tool_call>{"name": "y", "parameters": {"b": 2}}</tool_call>')
    assert errors == [] and calls[0].arguments == {"a": 1} and calls[1].arguments == {"b": 2}


def test_parse_fenced_json_fallback():
    _, calls, _ = parse_tool_calls('```json\n{"name": "x", "arguments": {}}\n```')
    assert calls[0].name == "x"


def test_malformed_json_is_an_error_not_a_final_answer():
    resp = response_from_text('<tool_call>{"name": "search", "arguments": {"pattern": "def median}}</tool_call>')
    assert resp.tool_calls == [] and not resp.final
    assert "Unterminated string" in resp.parse_errors[0]


def test_unclosed_tag_still_parses():
    _, calls, _ = parse_tool_calls('<tool_call>{"name": "x", "arguments": {}}')
    assert calls[0].name == "x"


def test_plain_text_is_final():
    assert response_from_text("All done.").final


def test_feedback_message_names_the_problem():
    msg = format_call_errors(ToolCall("x", {}), ["missing required argument 'a'"])
    assert "'x' was rejected" in msg and "missing required argument" in msg
