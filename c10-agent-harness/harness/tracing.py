"""Tracing: where the time, the tokens and the failures went.

One Span per model call and per tool call. The summary answers the questions you
ask after a bad run: how many calls, how many tokens, which tool was slow, how
many retries, how many failures.
"""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Iterator


@dataclass
class Span:
    kind: str                 # "model" | "tool"
    name: str
    step: int
    duration_s: float = 0.0
    ok: bool = True
    prompt_tokens: int = 0
    completion_tokens: int = 0
    retries: int = 0
    error: str | None = None


class Tracer:
    def __init__(self) -> None:
        self.spans: list[Span] = []

    @contextmanager
    def span(self, kind: str, name: str, step: int) -> Iterator[Span]:
        s = Span(kind, name, step)
        t0 = time.perf_counter()
        try:
            yield s
        except Exception as e:
            s.ok, s.error = False, f"{type(e).__name__}: {e}"
            raise
        finally:
            s.duration_s = time.perf_counter() - t0
            self.spans.append(s)

    def summary(self) -> dict:
        model = [s for s in self.spans if s.kind == "model"]
        tools = [s for s in self.spans if s.kind == "tool"]
        by_tool: dict[str, list[Span]] = defaultdict(list)
        for s in tools:
            by_tool[s.name].append(s)
        return {
            "model_calls": len(model),
            "tool_calls": len(tools),
            "prompt_tokens": sum(s.prompt_tokens for s in model),
            "completion_tokens": sum(s.completion_tokens for s in model),
            "model_latency_s": round(sum(s.duration_s for s in model), 3),
            "tool_latency_s": round(sum(s.duration_s for s in tools), 3),
            "retries": sum(s.retries for s in self.spans),
            "failures": sum(not s.ok for s in self.spans),
            "by_tool": {k: {"calls": len(v), "failures": sum(not s.ok for s in v),
                            "latency_s": round(sum(s.duration_s for s in v), 3)}
                        for k, v in sorted(by_tool.items())},
        }

    def table(self) -> str:
        """A plain-text report for terminals and READMEs."""
        s = self.summary()
        rows = [f"model calls     {s['model_calls']:>6}   latency {s['model_latency_s']:>8.3f}s",
                f"tool calls      {s['tool_calls']:>6}   latency {s['tool_latency_s']:>8.3f}s",
                f"prompt tokens   {s['prompt_tokens']:>6}   completion tokens {s['completion_tokens']}",
                f"retries         {s['retries']:>6}   failures {s['failures']}",
                "", f"{'tool':<22}{'calls':>6}{'fail':>6}{'latency':>10}"]
        rows += [f"{k:<22}{v['calls']:>6}{v['failures']:>6}{v['latency_s']:>9.3f}s"
                 for k, v in s["by_tool"].items()]
        return "\n".join(rows)

    def to_list(self) -> list[dict]:
        return [asdict(s) for s in self.spans]
