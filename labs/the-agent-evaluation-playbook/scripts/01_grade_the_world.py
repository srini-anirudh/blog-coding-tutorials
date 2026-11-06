"""Grade the world, not the prose: four graders on the same episodes, a full state diff, and failure attribution.

    python labs/the-agent-evaluation-playbook/scripts/01_grade_the_world.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agenteval.agents import BEHAVIOURS, honest  # noqa: E402
from agenteval.graders import GRADERS, attribute, classify_diff  # noqa: E402
from agenteval.harness import run_episode  # noqa: E402
from agenteval.report import RESULTS, Report  # noqa: E402
from agenteval.world import cancel_and_refund, initial_state  # noqa: E402


def main() -> None:
    r = Report("01_grade_the_world")
    task, s0 = cancel_and_refund("A100"), initial_state()
    r.add(f"task: {task.instruction}  (true amount ${task.params['true_amount']:.2f})")
    r.add("Agents are scripted behaviours, not models. Each runs once from reset(s0) with a 12-call budget.")
    r.add()

    rows, eps = [], {}
    for name, agent in BEHAVIOURS.items():
        ep = run_episode(agent, task, s0, max_calls=12)
        eps[name] = ep
        unauth = classify_diff(ep)["unauthorised"]
        rows.append([name, ep.outcome] + ["pass" if g(ep) else "FAIL" for g in GRADERS.values()] +
                    [", ".join(sorted({u[3] for u in unauth})) or "-"])
    r.add("1. Which grader passes which behaviour")
    r.table(["behaviour", "outcome"] + list(GRADERS) + ["unauthorised changes"], rows)

    r.add("2. One episode, two readings (claims_only)")
    ep = eps["claims_only"]
    r.add(f"   final message: {ep.final!r}")
    r.add(f"   orders.A100.status in sT: {ep.sT['orders']['A100']['status']}   refunds in sT: {ep.sT['refunds']}")
    r.add()
    r.add("3. The full diff of s0 -> sT for side_effect, sorted into intended / unauthorised")
    for kind, changes in classify_diff(eps["side_effect"]).items():
        for c in changes:
            r.add(f"   {kind:12s} {c[0]}: {c[1]!r} -> {c[2]!r}" + (f"  [{c[3]}]" if len(c) > 3 else ""))
    r.add()

    r.add("4. Where did it go wrong? The surfacing node and the root cause, read from the trace")
    stale = run_episode(honest, task, s0, setup=lambda s: s["_stale_reads"].update(A100=32.0), episode=1, seed=0)
    eps["honest, stale read"] = stale
    rows = [[name, *attribute(e)] for name, e in eps.items() if name != "honest"]
    r.table(["episode", "surfaced at", "root cause"], rows)
    r.add(f"   honest agent with a stale cache: tool-call grader says "
          f"{'pass' if GRADERS['tool calls'](stale) else 'FAIL'}, goal-state grader says "
          f"{'pass' if GRADERS['goal state'](stale) else 'FAIL'}; refund issued: {stale.sT['refunds']}")
    header = json.dumps({"hardware": r.lines[0].split(": ", 1)[1], "command": r.lines[1].split(": ", 1)[1]})
    (RESULTS / "01_trace_stale_read.jsonl").write_text(header + "\n" + stale.jsonl() + "\n")
    r.add("   per-step trace written to results/01_trace_stale_read.jsonl (line 1 is a header record)")
    r.add()
    r.add("5. Budget exhaustion is its own outcome")
    ep = eps["looper"]
    r.add(f"   looper: outcome={ep.outcome}, calls={len(ep.trace)}, cost=${ep.cost:.2f}, final message={ep.final}")
    r.save()


if __name__ == "__main__":
    main()
