"""The tool registry: Python functions in, JSON schemas and validated calls out.

    @tool(risk=Risk.READ)
    def read_file(ctx: ToolContext, path: str, start: int = 1, end: int | None = None) -> str:
        '''Read a text file with line numbers.

        Args:
            path: File path relative to the workspace root.
            start: First line to show (1-based).
            end: Last line to show; omit for the end of the file.
        '''

The schema is generated from the signature and the docstring, so the description
the model sees can never drift from the function that runs.
"""

from __future__ import annotations

import enum
import inspect
import re
import types
import typing
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Union, get_args, get_origin

from .messages import ToolCall
from .permissions import Risk


@dataclass
class ToolOutput:
    """What a tool hands back. `kind` lets the event log emit Patch / TestResult events."""

    content: str
    is_error: bool = False
    kind: str | None = None
    data: dict[str, Any] = field(default_factory=dict)


class ToolError(Exception):
    """Raise inside a tool for an expected failure; the message goes to the model verbatim."""


@dataclass
class ToolContext:
    """Injected into any tool parameter annotated `ToolContext`; never shown to the model."""

    workspace: Any                      # pathlib.Path of the workspace root
    sandbox: Any = None                 # harness.sandbox.Sandbox used for commands
    files: Any = None                   # harness.fs_tools.FileTracker (read-before-write)
    plan: Any = None                    # harness.planning.Plan
    extras: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------- schema generation

_PRIMITIVES: dict[Any, dict[str, Any]] = {
    str: {"type": "string"}, int: {"type": "integer"}, float: {"type": "number"},
    bool: {"type": "boolean"}, dict: {"type": "object"}, list: {"type": "array"},
}


def type_to_schema(tp: Any) -> dict[str, Any]:
    """Translate one type hint into a JSON-schema fragment."""
    if tp in _PRIMITIVES:
        return dict(_PRIMITIVES[tp])
    if inspect.isclass(tp) and issubclass(tp, enum.Enum):
        return {"type": "string", "enum": [m.value for m in tp]}
    origin, args = get_origin(tp), get_args(tp)
    if origin is Literal:
        kinds = {type(a) for a in args}
        base = type_to_schema(kinds.pop()) if len(kinds) == 1 else {}
        return {**base, "enum": list(args)}
    if origin in (Union, types.UnionType):
        non_null = [a for a in args if a is not type(None)]
        if len(non_null) == 1:            # Optional[X]: nullability is expressed by "not required"
            return type_to_schema(non_null[0])
        return {"anyOf": [type_to_schema(a) for a in non_null]}
    if origin in (list, tuple, set, frozenset):
        return {"type": "array", "items": type_to_schema(args[0])} if args else {"type": "array"}
    if origin is dict:
        return {"type": "object"}
    raise TypeError(f"no JSON schema for type hint {tp!r}")


def parse_docstring(doc: str | None) -> tuple[str, dict[str, str]]:
    """Split a Google-style docstring into (summary, {param: description})."""
    if not doc:
        return "", {}
    doc = inspect.cleandoc(doc)
    head, _, rest = doc.partition("\nArgs:\n")
    summary = " ".join(head.strip().split("\n\n")[0].split())
    params: dict[str, str] = {}
    current = None
    for line in rest.splitlines():
        if re.match(r"^\S", line):          # a dedented line ends the Args block
            break
        m = re.match(r"^\s+(\w+)(?:\s*\(.*?\))?:\s*(.*)$", line)
        if m:
            current = m.group(1)
            params[current] = m.group(2).strip()
        elif current and line.strip():
            params[current] += " " + line.strip()
    return summary, params


@dataclass
class Tool:
    name: str
    fn: Callable[..., Any]
    description: str
    parameters: dict[str, Any]          # JSON schema of the arguments object
    risk: Risk | Callable[[dict], Risk]
    context_param: str | None = None    # parameter that receives the ToolContext

    def schema(self) -> dict[str, Any]:
        """OpenAI / Hugging Face function-tool format."""
        return {"type": "function",
                "function": {"name": self.name, "description": self.description,
                             "parameters": self.parameters}}

    def risk_for(self, arguments: dict) -> Risk:
        return self.risk(arguments) if callable(self.risk) else self.risk


def build_tool(fn: Callable[..., Any], name: str | None = None,
               risk: Risk | Callable[[dict], Risk] = Risk.READ) -> Tool:
    """Inspect `fn` and build its Tool, including the JSON schema."""
    hints = typing.get_type_hints(fn)
    summary, param_docs = parse_docstring(fn.__doc__)
    props: dict[str, Any] = {}
    required: list[str] = []
    context_param = None
    for pname, p in inspect.signature(fn).parameters.items():
        hint = hints.get(pname, str)
        if hint is ToolContext:
            context_param = pname
            continue
        prop = type_to_schema(hint)
        if pname in param_docs:
            prop["description"] = param_docs[pname]
        if p.default is inspect.Parameter.empty:
            required.append(pname)
        elif p.default is not None:
            prop["default"] = p.default.value if isinstance(p.default, enum.Enum) else p.default
        props[pname] = prop
    params = {"type": "object", "properties": props, "required": required,
              "additionalProperties": False}
    return Tool(name or fn.__name__, fn, summary, params, risk, context_param)


def tool(fn: Callable[..., Any] | None = None, *, name: str | None = None,
         risk: Risk | Callable[[dict], Risk] = Risk.READ):
    """Decorator that attaches a Tool to a function (`fn.__tool__`). Works with or without ()."""
    def wrap(f: Callable[..., Any]) -> Callable[..., Any]:
        f.__tool__ = build_tool(f, name=name, risk=risk)  # type: ignore[attr-defined]
        return f
    return wrap(fn) if fn is not None else wrap


# ---------------------------------------------------------------- validation

_JSON_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,), "integer": (int,), "number": (int, float),
    "boolean": (bool,), "object": (dict,), "array": (list,),
}


def _coerce(value: Any, schema: dict[str, Any]) -> Any:
    """Small models often quote numbers ("12") or booleans ("true"). Accept those."""
    t = schema.get("type")
    if isinstance(value, str):
        if t == "integer" and re.fullmatch(r"-?\d+", value.strip()):
            return int(value)
        if t == "number" and re.fullmatch(r"-?\d+(\.\d*)?", value.strip()):
            return float(value)
        if t == "boolean" and value.lower() in ("true", "false"):
            return value.lower() == "true"
    return value


def check_value(value: Any, schema: dict[str, Any], where: str) -> list[str]:
    """Errors for one value against a schema fragment (the subset our generator emits)."""
    if "anyOf" in schema:
        if any(not check_value(value, s, where) for s in schema["anyOf"]):
            return []
        return [f"{where}: {value!r} matches none of the allowed types"]
    errors: list[str] = []
    t = schema.get("type")
    if t is not None:
        ok = isinstance(value, _JSON_TYPES[t]) and not (t in ("integer", "number") and isinstance(value, bool))
        if not ok:
            return [f"{where}: expected {t}, got {type(value).__name__} {value!r}"]
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{where}: {value!r} is not one of {schema['enum']}")
    if t == "array" and "items" in schema:
        for i, item in enumerate(value):
            errors += check_value(item, schema["items"], f"{where}[{i}]")
    return errors


# ---------------------------------------------------------------- registry

class ToolRegistry:
    """Holds the tools the model may call; validates and executes calls."""

    def __init__(self, tools: list[Callable[..., Any] | Tool] | None = None, coerce: bool = True):
        self._tools: dict[str, Tool] = {}
        self.coerce = coerce
        for t in tools or []:
            self.register(t)

    def register(self, fn_or_tool: Callable[..., Any] | Tool) -> Tool:
        t = fn_or_tool if isinstance(fn_or_tool, Tool) else getattr(fn_or_tool, "__tool__", None)
        if t is None:
            t = build_tool(fn_or_tool)  # plain function: register with default READ risk
        if t.name in self._tools:
            raise ValueError(f"duplicate tool name {t.name!r}")
        self._tools[t.name] = t
        return t

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __getitem__(self, name: str) -> Tool:
        return self._tools[name]

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        return [t.schema() for t in self._tools.values()]

    def validate(self, call: ToolCall) -> list[str]:
        """Return a list of human-readable problems; empty means the call may run.

        Coerced values are written back into `call.arguments`.
        """
        if call.name not in self._tools:
            return [f"unknown tool {call.name!r}; available tools: {', '.join(self.names)}"]
        if not isinstance(call.arguments, dict):
            return [f"arguments must be a JSON object, got {type(call.arguments).__name__}"]
        params = self._tools[call.name].parameters
        props = params["properties"]
        errors = [f"missing required argument {r!r}" for r in params["required"] if r not in call.arguments]
        for key, value in list(call.arguments.items()):
            if key not in props:
                errors.append(f"unexpected argument {key!r}; allowed: {', '.join(props) or 'none'}")
                continue
            if self.coerce:
                value = call.arguments[key] = _coerce(value, props[key])
            errors += check_value(value, props[key], key)
        return errors

    def execute(self, call: ToolCall, ctx: ToolContext | None = None) -> ToolOutput:
        """Run a validated call. Exceptions become error outputs, never crashes."""
        t = self._tools[call.name]
        kwargs = dict(call.arguments)
        if t.context_param:
            kwargs[t.context_param] = ctx
        try:
            out = t.fn(**kwargs)
        except ToolError as e:
            return ToolOutput(f"Error: {e}", is_error=True)
        except Exception as e:  # a tool bug must not kill the loop; the model sees a short report
            return ToolOutput(f"Error: {type(e).__name__}: {e}", is_error=True)
        return out if isinstance(out, ToolOutput) else ToolOutput(str(out))
