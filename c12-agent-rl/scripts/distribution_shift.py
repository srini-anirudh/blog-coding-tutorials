"""Measure distribution shift: success against the number of injected mistakes, and how often the policy acts in a
state no demonstration contains.

    python c12-agent-rl/scripts/distribution_shift.py --ckpt c12-agent-rl/runs/bc.pt --out shift_bc
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agentrl.bc import expert_demos  # noqa: E402
from agentrl.codec import WordCodec  # noqa: E402
from agentrl.env import FileSystemEnv, run_episode  # noqa: E402
from agentrl.evaluate import Episode, run_suite  # noqa: E402
from agentrl.expert import ScriptedExpert  # noqa: E402
from agentrl.policy import TinyPolicy  # noqa: E402
from agentrl.shift import perturbations, seen_signatures, signature, unseen_rate  # noqa: E402
from agentrl.tasks import sample_task  # noqa: E402


def perturbed_suite(k: int, n: int) -> list[Episode]:
    return [Episode(1 + i % 3, i, perturbations(sample_task(1 + i % 3, i), k, i)) for i in range(n)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "runs/bc.pt"))
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", default="shift_bc")
    args = ap.parse_args()
    torch.set_num_threads(2)
    codec = WordCodec()
    policy = TinyPolicy.load(args.ckpt).eval()
    seen = seen_signatures(expert_demos(5000, (1, 2, 3), codec))
    print(f"{len(seen)} distinct abstract states in 5000 demonstrations")

    rows = []
    print("mistakes  success  expert  unseen-state rate  submitted")
    for k in range(4):
        eps = perturbed_suite(k, args.n)
        trajs = run_suite(policy, codec, eps, device="cpu")
        expert = 0
        for e in eps:
            task = sample_task(e.task_level, e.seed)
            expert += run_episode(FileSystemEnv(task), ScriptedExpert(task.text), forced=e.forced).success
        row = {"k": k, "success": sum(t.success for t in trajs) / args.n, "expert": expert / args.n,
               "unseen": unseen_rate(trajs, seen), "submitted": sum(t.submitted for t in trajs) / args.n}
        rows.append(row)
        print(f"{k:8d}  {row['success']:7.3f}  {row['expert']:6.3f}  {row['unseen']:17.3f}  {row['submitted']:9.3f}")

    examples = []
    trajs = run_suite(policy, codec, perturbed_suite(1, args.n), device="cpu")
    failed = [t for t in trajs if not t.success][:3]
    for t in failed:
        print()
        print(t.pretty())
        for i, s in enumerate(t.steps):
            print(f"   step {i}: state seen in demos = {signature(t, i) in seen}")
        examples.append({"task": t.task_text, "steps": [[s.action, s.observation, s.forced] for s in t.steps],
                         "seen": [signature(t, i) in seen for i in range(t.n_steps)]})
    (ROOT / "results" / f"{args.out}.json").write_text(json.dumps({"rows": rows, "examples": examples}, indent=1))


if __name__ == "__main__":
    main()
