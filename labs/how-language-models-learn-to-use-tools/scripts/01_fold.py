"""A tool call is only tokens: the agent loop as a fold over contexts.

Part 1 runs a scripted model through three steps, one of which is a mistake, and prints the context as it
grows. Part 2, if a checkpoint from 03_tool_diversity.py exists, lets the trained tiny model emit a call:
the harness parses it and runs it. Without the harness the same tokens would be inert text.

    python labs/how-language-models-learn-to-use-tools/scripts/01_fold.py
"""

from __future__ import annotations

import random
import re
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))

from toollab.hw import header  # noqa: E402
from toollab.loop import fold  # noqa: E402
from toollab.world import Vocab, encode_result, execute, make_example, make_tools, parse_call, split_names  # noqa: E402

CALL = re.compile(r"<tool_call>(\w+)\((.*?)\)</tool_call>")


def parse_text(span: str) -> dict | None:
    m = CALL.search(span)
    if not m:
        return None
    args = dict(kv.split("=", 1) for kv in m.group(2).split(",") if kv)
    return {"name": m.group(1), "args": args}


def text_backend(call: dict) -> str:
    tables = {"weather": {"malmo": "4C, rain", "oslo": "-2C, snow"},
              "convert": {"4": "39.2F", "-2": "28.4F"}}
    if call["name"] == "weather":
        city = call["args"]["city"]
        if city not in tables["weather"]:
            raise KeyError(f"unknown city {city!r}")
        return tables["weather"][city]
    if call["name"] == "convert":
        return tables["convert"][call["args"]["celsius"]]
    raise KeyError(f"no such tool {call['name']!r}")


def scripted_model():
    """Stands in for p_theta. It reads only the context string it is given, like a real model."""
    replies = iter([
        "<tool_call>weather(city=malmoe)</tool_call>",  # a typo: the mistake stays in the context
        "<tool_call>weather(city=malmo)</tool_call>",  # it reads the error and retries
        "<tool_call>convert(celsius=4)</tool_call>",
        "It is 4C (39.2F) and raining in Malmo.",
    ])
    return lambda context: next(replies)


def main() -> None:
    header()
    x0 = "<user> What is the weather in Malmo, in Fahrenheit?\n<assistant> "
    tr = fold(scripted_model(), x0, parse_text, text_backend, lambda r: f"\n<result>{r}</result>\n")
    print("Part 1: a scripted model in the fold\n")
    for k, s in enumerate(tr.steps):
        print(f"step {k}: context {s.context_len:>3} chars | model emitted {s.span!r}")
        print(f"        Parse -> {s.call}")
        print(f"        harness wrote back {s.observation!r}")
    print("\nFinal context (the agent's entire state):\n" + tr.context)
    print("\nThe same first span with no harness is just a string. Nothing ran:", repr(tr.steps[0].span))

    for n in (16, 1024):
        ck = LAB / "runs" / f"tools_N{n}.pt"
        if not ck.exists():
            print(f"\nPart 2 skipped for N = {n}: run scripts/03_tool_diversity.py first.")
            continue
        part2(ck)


def part2(ck: Path) -> None:
    from toollab.model import decode, load

    model, meta = load(ck)
    vocab = Vocab()
    _, heldout = split_names(seed=0)
    pool = make_tools(300, heldout, random.Random(7))
    print(f"\nPart 2: the tiny model trained on N = {meta['n_tools']} tools, on tools it never saw")
    for seed in range(5, 11):  # six fixed requests, not picked by outcome
        ex = make_example(pool, random.Random(seed), n_tools=3, p_none=0.0)
        span = decode(model, vocab, ex)
        print("\ncontext x_0:", " ".join(ex.prompt_tokens()))
        print("model span :", " ".join(span))
        call = parse_call(span)
        print("Parse      :", call)
        try:
            obs = encode_result(execute(call, ex.tools)) if call and not call.get("none") else "no call"
        except Exception as err:
            obs = encode_result(f"error: {err}")
        print("Exec, e    :", obs)
        print("gold call  :", " ".join(ex.target_tokens()))

if __name__ == "__main__":
    main()
