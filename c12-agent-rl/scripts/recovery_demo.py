"""The same failure-injection tasks before and after RL, greedy. Writes results/recovery.txt and results/recovery.json.

    python c12-agent-rl/scripts/recovery_demo.py

Recovery rate = success among episodes where at least one tool call returned an error.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agentrl.codec import WordCodec  # noqa: E402
from agentrl.evaluate import run_suite, suite  # noqa: E402
from agentrl.failures import INVALID, MISSING, TIMEOUT, TOOL_ERROR  # noqa: E402
from agentrl.policy import TinyPolicy  # noqa: E402


def recovery(trajs) -> dict:
    hit = [t for t in trajs if any(s.error for s in t.steps)]
    by_type: Counter = Counter()
    won: Counter = Counter()
    for t in hit:
        kinds = {s.observation for s in t.steps if s.error}
        for k in kinds & {TIMEOUT, TOOL_ERROR, INVALID, MISSING}:
            by_type[k] += 1
            won[k] += t.success
    return {"episodes": len(trajs), "hit_error": len(hit), "recovered": sum(t.success for t in hit),
            "by_type": {k: [won[k], by_type[k]] for k in by_type}}


def main() -> None:
    codec = WordCodec()
    eps = suite("failures", 300)
    out, payload = [], {}
    runs = {}
    for name in ("bc", "grpo"):
        policy = TinyPolicy.load(str(ROOT / "runs" / f"{name}.pt")).eval()
        runs[name] = run_suite(policy, codec, eps, device="cpu")
        payload[name] = recovery(runs[name])
        p = payload[name]
        out.append(f"{name:<5} success {sum(t.success for t in runs[name]) / len(eps):.3f}  "
                   f"recovered {p['recovered']}/{p['hit_error']} episodes with an error")
        for k, (w, n) in sorted(p["by_type"].items()):
            out.append(f"        {k:<22} {w}/{n}")
    # one task the cloned policy loses and the trained policy wins
    pair = next(((a, b) for a, b in zip(runs["bc"], runs["grpo"]) if not a.success and b.success
                 and any(s.error for s in b.steps) and any(s.error for s in a.steps)), None)
    if pair:
        out += ["", "before RL (bc):", pair[0].pretty(), "", "after RL (grpo):", pair[1].pretty()]
    text = "\n".join(out)
    print(text)
    (ROOT / "results" / "recovery.txt").write_text(text + "\n")
    (ROOT / "results" / "recovery.json").write_text(json.dumps(payload, indent=1))


if __name__ == "__main__":
    main()
