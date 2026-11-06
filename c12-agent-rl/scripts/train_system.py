"""Run the full pipeline: task sampler, env workers, rollouts, store, verifier, batches, update.

    python c12-agent-rl/scripts/train_system.py --iters 10

Runs three configurations from the BC checkpoint: instant tools, 20 ms tools inline, 20 ms tools with 32 workers.
Prints the time spent in each box of the pipeline. Writes results/system.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agentrl.codec import WordCodec  # noqa: E402
from agentrl.evaluate import evaluate  # noqa: E402
from agentrl.grpo import GRPOConfig  # noqa: E402
from agentrl.online import TaskMix, TaskSampler  # noqa: E402
from agentrl.policy import TinyPolicy  # noqa: E402
from agentrl.system import AgentRLSystem, SystemConfig  # noqa: E402


def run(args, workers: int, latency: float, codec) -> dict:
    torch.manual_seed(args.seed)
    policy = TinyPolicy.load(str(ROOT / "runs" / "bc.pt"))
    store = ROOT / "runs" / f"store_w{workers}_l{int(latency * 1000)}.jsonl"
    store.unlink(missing_ok=True)
    system = AgentRLSystem(policy, codec, TaskSampler(TaskMix({1: 1, 2: 1, 3: 1, 4: 2, 5: 2}, 0.5), seed=args.seed),
                           GRPOConfig(), SystemConfig(args.tasks, args.G, workers=workers, tool_latency=latency),
                           store)
    clock: dict[str, float] = defaultdict(float)
    for _ in range(args.iters):
        row = system.iteration()
        for k, v in row["clock"].items():
            clock[k] += v
        print(f"v{row['version']:<3d} success {row['success']:.2f}  zero-var {row['zero_var_groups']:.2f}  "
              + "  ".join(f"{k} {v:.2f}s" for k, v in row["clock"].items()), flush=True)
    system.save(ROOT / "runs" / "system.pt")
    total = sum(clock.values())
    print(f"workers={workers} latency={1000 * latency:.0f}ms  total {total:.1f}s  " +
          "  ".join(f"{k} {100 * v / total:.1f}%" for k, v in clock.items()), flush=True)
    print(f"store: {len(system.store.load())} trajectories, {store.stat().st_size / 1e6:.2f} MB; "
          f"verifier: {system.verifier.checked} checked, {system.verifier.mismatches} mismatches", flush=True)
    return {"workers": workers, "latency_ms": 1000 * latency, "total": total, "clock": dict(clock),
            "stored": len(system.store.load()), "store_mb": store.stat().st_size / 1e6,
            "checked": system.verifier.checked, "mismatches": system.verifier.mismatches,
            "eval": evaluate(policy, codec, device="cpu", n=150)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--tasks", type=int, default=24)
    ap.add_argument("--G", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    codec = WordCodec()
    out = [run(args, 0, 0.0, codec), run(args, 0, 0.02, codec), run(args, 32, 0.02, codec)]
    (ROOT / "results" / "system.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
