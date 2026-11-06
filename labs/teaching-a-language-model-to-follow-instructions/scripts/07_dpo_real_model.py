"""Likelihood displacement on a real model: DPO on SmolLM2-135M-Instruct, with near-identical chosen and rejected
responses (a correct arithmetic solution versus the same text with one number wrong). Downloads the model once.

    python labs/teaching-a-language-model-to-follow-instructions/scripts/07_dpo_real_model.py --steps 60
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prefopt.common import Report  # noqa: E402
from prefopt.lm import MODEL, build_examples, load, train  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--pairs", type=int, default=400)
    ap.add_argument("--methods", default="dpo,dpop,nll")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    torch.manual_seed(0)
    device = torch.device(args.device)
    rep = Report("07_dpo_real_model" + ("_gpu" if device.type == "cuda" else ""), device)
    rep.print(f"{MODEL}, {args.pairs} synthetic pairs, batch {args.batch}, lr {args.lr}, beta {args.beta}, "
              f"{args.steps} steps. Measured on the first 32 training pairs.")
    rep.print("d_logp = mean change in summed log pi(y | x) from the reference, in nats.")
    results = {}
    for method in args.methods.split(","):
        tok, policy = load(device)
        examples = build_examples(tok, args.pairs)
        if method == args.methods.split(",")[0]:
            ex = examples[0]
            same = sum(a == b for a, b in zip(ex["chosen_ids"], ex["rejected_ids"]))
            rep.print(f"example pair: {len(ex['chosen_ids'])} chosen tokens, {len(ex['rejected_ids'])} rejected, "
                      f"{same} identical at the same position.")
        rep.print(f"\n{method}")
        hist = train(policy, tok, examples, device, method=method, steps=args.steps, batch=args.batch, lr=args.lr,
                     beta=args.beta, log=lambda row: None)
        rep.table([{k: v for k, v in r.items() if k != "sec"} for r in hist], "{:.3f}")
        rep.print(f"wall time {hist[-1]['sec']:.0f} s")
        results[method] = hist
    rep.data.update({"model": MODEL, "args": vars(args), "runs": results})
    rep.save()


if __name__ == "__main__":
    main()
