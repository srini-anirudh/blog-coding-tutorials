"""The 8-line loop: one good run, then the three ways it breaks.

    python c10-agent-harness/scripts/minimal_loop.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness import ScriptedModel, call  # noqa: E402
from harness.loop_minimal import run_minimal  # noqa: E402
from harness.models import ScriptExhausted  # noqa: E402

TOOLS = {"add": lambda a, b: a + b}


def main() -> None:
    print("--- a good run ---")
    context = run_minimal(ScriptedModel([call("add", a=2, b=3), "2 + 3 = 5."]), TOOLS, "What is 2 + 3?")
    for m in context:
        calls = [f'{c["function"]["name"]}({c["function"]["arguments"]})' for c in m.get("tool_calls", [])]
        print(f'{m["role"]:<10} {m["content"]!r} {calls or ""}')

    print("\n--- break 1: the model names a tool that does not exist ---")
    try:
        run_minimal(ScriptedModel([call("multiply", a=2, b=3)]), TOOLS, "What is 2 * 3?")
    except KeyError as e:
        print(f"the whole run dies: KeyError {e}")

    print("\n--- break 2: the arguments do not fit the function ---")
    try:
        run_minimal(ScriptedModel([call("add", a=2)]), TOOLS, "What is 2 + ?")
    except TypeError as e:
        print(f"the whole run dies: TypeError {e}")

    print("\n--- break 3: a malformed call (a real model would repeat this forever) ---")
    model = ScriptedModel(['<tool_call>{"name": "add", "arguments": {"a": 2,}}</tool_call>'] * 5)
    try:
        run_minimal(model, TOOLS, "What is 2 + 3?")
    except ScriptExhausted:
        print(f"no tool ran, no error reached the model, and the loop called it {model.calls} times "
              "until the script ran out")


if __name__ == "__main__":
    main()
