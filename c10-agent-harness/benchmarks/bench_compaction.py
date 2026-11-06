"""Why the loop is quadratic, and what compaction does about it.

Every model call re-sends the whole history, so total prompt tokens grow with the
square of the number of steps. We run a scripted session of N tool calls, each
returning a fixed-size observation, with and without a context budget.

    python c10-agent-harness/benchmarks/bench_compaction.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from harness import Agent, AgentConfig, ContextBuilder, ScriptedModel, ToolRegistry, call, tool  # noqa: E402


@tool
def inspect_file(i: int) -> str:
    """Return a file-sized observation.

    Args:
        i: Which file.
    """
    return f"file {i}\n" + "\n".join(f"line {k}: x = compute(x, {k})" for k in range(30))


def run(steps: int, budget: int | None) -> dict:
    script = [call("inspect_file", i=i) for i in range(steps)] + ["done"]
    cfg = AgentConfig(max_steps=steps + 1, checkpoint_on_start=False,
                      context_budget=budget or 10**9, keep_recent_turns=4)
    with tempfile.TemporaryDirectory() as tmp:
        agent = Agent(ScriptedModel(script), ToolRegistry([inspect_file]), tmp, context=ContextBuilder(), config=cfg)
        result = agent.run("Inspect every file.")
    assert result.status == "done"
    per_step = [e.data["usage"]["prompt"] for e in agent.log.of_type("ModelResponse")]
    replay_ok = agent.log.to_messages() == [m for _, m in agent.history]
    return {"steps": steps, "budget": budget, "cumulative": sum(per_step), "peak": max(per_step),
            "compactions": len(agent.log.of_type("Compaction")), "per_step": per_step, "replay_ok": replay_ok}


def main() -> None:
    print(f"{'steps':>6}{'budget':>10}{'peak prompt':>14}{'cumulative':>13}{'compactions':>13}{'saved':>8}")
    series = {}
    for steps in (10, 20, 40, 80, 160):
        base = None
        for budget in (None, 4000, 2000):
            r = run(steps, budget)
            base = base or r["cumulative"]
            print(f"{steps:>6}{budget or 'none':>10}{r['peak']:>14}{r['cumulative']:>13}{r['compactions']:>13}"
                  f"{1 - r['cumulative'] / base:>8.0%}")
            assert r["replay_ok"]
            if steps == 80:
                series[str(budget)] = r["per_step"]
    out = HERE / "results" / "compaction_series.json"
    out.write_text(json.dumps(series))
    print("\nper-step prompt sizes for the 80-step run saved to results/compaction_series.json")


if __name__ == "__main__":
    main()
