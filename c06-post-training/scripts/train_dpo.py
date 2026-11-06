"""Stage 2: DPO on the SFT model, with our manual loss.

    python scripts/train_dpo.py --steps 100 --pairs on-policy
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from posttrain.chat import load_tokenizer  # noqa: E402
from posttrain.dpo import collate_pairs, dpo_batch_loss, on_policy_pairs, synthetic_pairs, tokenize_pair  # noqa: E402
from posttrain.evaluate import evaluate  # noqa: E402
from posttrain.models import frozen_copy, load_policy  # noqa: E402
from posttrain.tasks import split_problems  # noqa: E402
from posttrain.utils import hardware_label, pick_device, polite_threads, save_json  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(ROOT / "checkpoints" / "sft-full"))
    ap.add_argument("--pairs", choices=["synthetic", "on-policy"], default="on-policy")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--n-problems", type=int, default=800)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    polite_threads()
    torch.manual_seed(0)
    device = pick_device()
    tok = load_tokenizer()
    policy = load_policy(args.base, device)
    ref = frozen_copy(policy)
    train, test = split_problems(args.n_problems, 200, seed=1)

    if args.pairs == "synthetic":
        pairs = synthetic_pairs(train)
    else:
        pairs = on_policy_pairs(policy, tok, train)
        print(f"on-policy: {len(pairs)} usable pairs from {len(train)} prompts")
    examples = [tokenize_pair(tok, p) for p in pairs]

    before = evaluate(policy, tok, test)
    print("before", before, flush=True)
    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=0.0)
    rng = random.Random(0)
    policy.train()
    history, t0 = [], time.time()
    for step in range(args.steps):
        chunk = rng.sample(examples, args.batch_size)
        batch = {k: v.to(device) for k, v in collate_pairs(chunk, tok.pad_token_id, multiple=32).items()}
        loss, m = dpo_batch_loss(policy, ref, batch, beta=args.beta)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if (step + 1) % 10 == 0 or step == 0:
            row = {"step": step + 1, "loss": round(loss.item(), 4), "margin": round(m["margin"].mean().item(), 3),
                   "pair_acc": round(m["accuracy"].mean().item(), 3), "sec": round(time.time() - t0, 1)}
            history.append(row)
            print(row, flush=True)
    after = evaluate(policy, tok, test)
    print("after", after, flush=True)

    tag = args.tag or f"dpo-{args.pairs}"
    policy.save_pretrained(ROOT / "checkpoints" / tag)
    tok.save_pretrained(ROOT / "checkpoints" / tag)
    save_json({"tag": tag, "hardware": hardware_label(device), "pairs": args.pairs, "n_pairs": len(pairs), "beta": args.beta,
               "lr": args.lr, "steps": args.steps, "before": before, "after": after, "history": history},
              f"{tag}.json")


if __name__ == "__main__":
    main()
