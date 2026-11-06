"""Stage 3 (optional): a pairwise reward model trained on preference pairs, scored on held-out pairs.

    python scripts/train_reward_model.py --steps 120
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
from posttrain.dpo import collate_pairs, synthetic_pairs, tokenize_pair  # noqa: E402
from posttrain.reward_model import RewardModel, pairwise_loss  # noqa: E402
from posttrain.tasks import split_problems  # noqa: E402
from posttrain.utils import pick_device, polite_threads, save_json  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


@torch.no_grad()
def held_out_accuracy(model, tok, examples, device, batch_size: int = 16) -> float:
    model.eval()
    hits = total = 0
    for s in range(0, len(examples), batch_size):
        b = collate_pairs(examples[s : s + batch_size], tok.pad_token_id, multiple=32)
        scores = model(b["input_ids"].to(device), b["attention_mask"].to(device))
        chosen, rejected = scores.chunk(2)
        hits += int((chosen > rejected).sum())
        total += chosen.numel()
    model.train()
    return hits / total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(ROOT / "checkpoints" / "sft-full"))
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    args = ap.parse_args()

    polite_threads()
    torch.manual_seed(0)
    device = pick_device()
    tok = load_tokenizer()
    train, test = split_problems(1000, 200, seed=2)
    train_ex = [tokenize_pair(tok, p) for p in synthetic_pairs(train, seed=2)]
    test_ex = [tokenize_pair(tok, p) for p in synthetic_pairs(test, seed=3)]

    model = RewardModel(args.base).to(device)
    # Two groups: the head starts at zero and can move faster than the pretrained body.
    opt = torch.optim.AdamW([{"params": model.body.parameters(), "lr": args.lr},
                             {"params": model.head.parameters(), "lr": args.lr * 50}], weight_decay=0.0)
    before = held_out_accuracy(model, tok, test_ex, device)
    print(f"held-out pair accuracy before: {before:.3f}", flush=True)
    rng, history, t0 = random.Random(0), [], time.time()
    model.train()
    for step in range(args.steps):
        b = collate_pairs(rng.sample(train_ex, args.batch_size), tok.pad_token_id, multiple=32)
        scores = model(b["input_ids"].to(device), b["attention_mask"].to(device))
        chosen, rejected = scores.chunk(2)
        loss, acc = pairwise_loss(chosen, rejected)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if (step + 1) % 20 == 0 or step == 0:
            row = {"step": step + 1, "loss": round(loss.item(), 4), "train_acc": round(acc.item(), 3),
                   "sec": round(time.time() - t0, 1)}
            history.append(row)
            print(row, flush=True)
    after = held_out_accuracy(model, tok, test_ex, device)
    print(f"held-out pair accuracy after: {after:.3f}", flush=True)
    save_json({"device": str(device), "steps": args.steps, "lr": args.lr, "before": before, "after": after,
               "history": history}, "reward_model.json")


if __name__ == "__main__":
    main()
