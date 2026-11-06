"""Rollouts per second: one at a time, batched, and with two threaded workers. Also the stage split.

    python benchmarks/bench_rollout.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from posttrain.chat import load_tokenizer  # noqa: E402
from posttrain.models import load_policy  # noqa: E402
from posttrain.rollout import RolloutEngine  # noqa: E402
from posttrain.tasks import exact_reward, sample_problems  # noqa: E402
from posttrain.utils import pick_device, polite_threads, save_json, synchronize  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def run(engine: RolloutEngine, problems, chunk: int, device) -> dict[str, float]:
    # One warm-up chunk so MPS kernels are compiled before the clock starts.
    engine.collect(problems[:chunk])
    t0 = time.time()
    n = gen = rew = 0.0
    for s in range(0, len(problems), chunk):
        n += len(engine.collect(problems[s : s + chunk]))
        gen += engine.timings["generate_sec"]
        rew += engine.timings["reward_sec"]
    synchronize(device)
    sec = time.time() - t0
    return {"rollouts": n, "sec": sec, "rollouts_per_sec": n / sec, "generate_sec": gen, "reward_sec": rew}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(ROOT / "checkpoints" / "sft-full"))
    ap.add_argument("--rollouts", type=int, default=64)
    args = ap.parse_args()
    polite_threads()
    device = pick_device()
    tok = load_tokenizer()
    policy = load_policy(args.base, device).eval()
    G = 4
    problems = sample_problems(args.rollouts // G, seed=6)

    results = {}
    results["sequential (batch 4)"] = run(RolloutEngine(policy, tok, exact_reward, 1, G), problems, 1, device)
    results["batched (batch 64)"] = run(RolloutEngine(policy, tok, exact_reward, 1, G), problems, len(problems), device)
    two = RolloutEngine(policy, tok, exact_reward, 2, G)
    results["2 workers (batch 32 each)"] = run(two, problems, len(problems), device)
    two.sync(policy, 1)
    sync_ms = two.timings["sync_sec"] * 1000

    print(f"device={device}, {args.rollouts} rollouts, up to 48 new tokens each")
    print(f"{'engine':<28}{'sec':>8}{'rollouts/sec':>14}{'generate':>10}{'reward':>10}")
    for name, r in results.items():
        print(f"{name:<28}{r['sec']:>8.2f}{r['rollouts_per_sec']:>14.1f}{r['generate_sec']:>10.2f}{r['reward_sec']:>10.4f}")
    print(f"weight sync to 2 replicas: {sync_ms:.0f} ms")
    save_json({"device": str(device), "results": results, "sync_ms": sync_ms}, "bench_rollout.json")


if __name__ == "__main__":
    main()
