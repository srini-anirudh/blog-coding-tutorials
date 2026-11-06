"""Print a JSONL event log as a readable transcript.

    python -m harness.replay runs/demo/events.jsonl
"""

from __future__ import annotations

import sys

from .events import Event, EventLog


def _clip(text: str, n: int = 6) -> str:
    lines = text.rstrip().splitlines()
    more = f"\n      ... ({len(lines) - n} more lines)" if len(lines) > n else ""
    return "\n".join("      " + ln for ln in lines[:n]) + more


def _short(value: object, n: int = 40) -> str:
    text = repr(value)
    return text if len(text) <= n else text[: n - 3] + "..."


def format_event(e: Event) -> str:
    d = e.data
    head = f"[{e.seq:>3}] step {e.step:>2}  {e.type:<13}"
    if e.type == "UserMessage":
        return f"{head} {d.get('kind', 'task')}\n{_clip(d['content'], 4)}"
    if e.type == "ModelResponse":
        calls = ", ".join(c["name"] for c in d["tool_calls"]) or (
            "(malformed call)" if d["parse_errors"] else "(final answer)")
        extra = f"  parse errors: {len(d['parse_errors'])}" if d["parse_errors"] else ""
        ctx = sum(d.get("context", {}).values())
        return f"{head} {d['text'][:70]!r} -> {calls}{extra}  [context ~{ctx} tok]"
    if e.type == "ToolCall":
        args = ", ".join(f"{k}={_short(v)}" for k, v in d["arguments"].items())
        return f"{head} {d['name']}({args})"
    if e.type == "ToolResult":
        flag = "ERROR " if d["is_error"] else ""
        return f"{head} {flag}{d['name']}\n{_clip(d['content'])}"
    if e.type == "Approval":
        return f"{head} {d['name']} risk={d['risk']} allowed={d['allowed']} ({d['reason']})"
    if e.type == "Patch":
        return f"{head} {d['path']} via {d['format']}"
    if e.type == "TestResult":
        return f"{head} {d['check']} {d['status']} passed={d['passed']} failed={d['failed']}"
    if e.type == "Compaction":
        return (f"{head} dropped {d['dropped_messages']} messages, "
                f"{d['tokens_before']} -> {d['tokens_after']} tokens")
    return f"{head} {d}"


def main(argv: list[str]) -> None:
    for e in EventLog.load(argv[0]).events:
        print(format_event(e))


if __name__ == "__main__":
    main(sys.argv[1:])
