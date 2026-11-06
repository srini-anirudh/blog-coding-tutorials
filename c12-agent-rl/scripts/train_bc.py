"""Behaviour cloning from clean expert demonstrations. Saves runs/bc.pt and results/bc.json.

    python c12-agent-rl/scripts/train_bc.py --demos 30000 --steps 3000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agentrl.bc import expert_demos, train_bc  # noqa: E402
from agentrl.codec import WordCodec  # noqa: E402
from agentrl.evaluate import evaluate, pick_device  # noqa: E402
from agentrl.policy import PolicyConfig, TinyPolicy  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", type=int, default=30000)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="bc")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    codec = WordCodec()
    demos = expert_demos(args.demos, tuple(args.levels), codec, seed=args.seed)
    policy = TinyPolicy(PolicyConfig(codec.vocab_size, d_model=args.d_model, n_layers=args.layers,
                                     max_len=384)).to(device)
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"device={device} threads={torch.get_num_threads()} vocab={codec.vocab_size} params={n_params:,} demos={len(demos)}")
    t0 = time.time()
    history = train_bc(policy, demos, steps=args.steps, batch_size=args.batch_size, lr=args.lr, device=device,
                       pad=codec.pad, seed=args.seed, log_every=max(args.steps // 15, 1))
    seconds = time.time() - t0
    for step, loss in history:
        print(f"step {step:5d}  loss {loss:.4f}")
    print(f"trained in {seconds:.0f}s")
    res = evaluate(policy, codec, device=device, n=300)
    print("success (greedy, 300 unseen tasks per suite):", {k: round(v, 3) for k, v in res.items()})
    (ROOT / "runs").mkdir(exist_ok=True)
    (ROOT / "results").mkdir(exist_ok=True)
    policy.cpu().save(ROOT / "runs" / f"{args.out}.pt")
    (ROOT / "results" / f"{args.out}.json").write_text(json.dumps(
        {"args": vars(args), "params": n_params, "seconds": seconds, "history": history, "eval": res}, indent=1))


if __name__ == "__main__":
    main()
