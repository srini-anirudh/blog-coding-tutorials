"""Stages 4-7: REINFORCE or GRPO on arithmetic with an exact reward, driven by the rollout engine.

    python scripts/train_rl.py --algo grpo --steps 40
    python scripts/train_rl.py --algo reinforce --steps 40

Both spend the same number of rollouts per step. Accuracy is greedy, on prompts the policy never trains on.
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
from posttrain.evaluate import evaluate  # noqa: E402
from posttrain.generation import score_completions  # noqa: E402
from posttrain.grpo import GRPOLossConfig, grpo_loss  # noqa: E402
from posttrain.models import frozen_copy, load_policy  # noqa: E402
from posttrain.reinforce import reinforce_loss  # noqa: E402
from posttrain.rollout import RolloutEngine  # noqa: E402
from posttrain.tasks import exact_reward, split_problems  # noqa: E402
from posttrain.utils import pick_device, polite_threads, save_json  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=["grpo", "reinforce"], default="grpo")
    ap.add_argument("--base", default=str(ROOT / "checkpoints" / "sft-full"))
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--prompts", type=int, default=8)
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--inner", type=int, default=2, help="GRPO: gradient steps per batch of rollouts")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--beta", type=float, default=0.04)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    polite_threads()
    torch.manual_seed(args.seed)
    device = pick_device()
    tok = load_tokenizer()
    policy = load_policy(args.base, device)
    ref = frozen_copy(policy) if args.algo == "grpo" and args.beta > 0 else None
    train, test = split_problems(4000, 200, seed=5)
    rng = random.Random(args.seed)

    group = args.group_size if args.algo == "grpo" else 1
    n_prompts = args.prompts if args.algo == "grpo" else args.prompts * args.group_size  # same rollouts per step
    engine = RolloutEngine(policy, tok, exact_reward, num_workers=1, group_size=group)
    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=0.0)
    loss_cfg = GRPOLossConfig(beta=args.beta)

    history = []
    start = evaluate(policy, tok, test)
    print("start", start, flush=True)
    t0 = time.time()
    for step in range(args.steps):
        policy.eval()
        engine.sync(policy, step)
        rollouts = engine.collect(rng.sample(train, n_prompts))
        batch = {k: v.to(device) for k, v in engine.build_batch(rollouts).items()}
        policy.train()
        args_tensors = (batch["prompt_ids"], batch["prompt_mask"], batch["completion_ids"], batch["completion_mask"])

        stats: dict[str, float] = {}
        if args.algo == "grpo":
            with torch.no_grad():
                old = score_completions(policy, *args_tensors)  # the policy that sampled this data
                ref_logps = score_completions(ref, *args_tensors) if ref is not None else None
            for _ in range(args.inner):
                logps = score_completions(policy, *args_tensors)
                loss, stats = grpo_loss(logps, old, batch["advantages"], batch["completion_mask"], ref_logps, loss_cfg)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
        else:
            logps = score_completions(policy, *args_tensors)
            loss = reinforce_loss(logps, batch["completion_mask"], batch["rewards"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)

        row = {"step": step + 1, "reward": round(batch["rewards"].mean().item(), 4), "loss": round(loss.item(), 4),
               "truncated": round(sum(r.truncated for r in rollouts) / len(rollouts), 3),
               "sec": round(time.time() - t0, 1), **{k: round(v, 5) for k, v in stats.items()}}
        if (step + 1) % args.eval_every == 0:
            row["eval"] = evaluate(policy, tok, test)
        history.append(row)
        print(row, flush=True)

    tag = args.tag or args.algo
    policy.save_pretrained(ROOT / "checkpoints" / tag)
    save_json({"tag": tag, "algo": args.algo, "device": str(device), "steps": args.steps, "prompts": n_prompts,
               "group_size": group, "inner": args.inner, "lr": args.lr, "beta": args.beta, "start": start,
               "history": history}, f"{tag}.json")


if __name__ == "__main__":
    main()
