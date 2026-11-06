"""Image-conditioned DPO on SmolVLM-256M with programmatic hallucination negatives.

    python scripts/train_dpo.py --negatives balanced --steps 150
    python scripts/train_dpo.py --negatives biased   --steps 150

Both runs are scored on the SAME held-out balanced pairs, so that any gain there
needed the image: on an exchangeable set a blind scorer sits at 50%.
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

from vlmalign.device import pick_device, seed_everything, set_threads, sync  # noqa: E402
from vlmalign.dpo import dpo_loss, pair_batch, pair_logprobs, reference_logprobs  # noqa: E402
from vlmalign.evaluate import preference_accuracy, vqa_accuracy, vqa_items  # noqa: E402
from vlmalign.freeze import configure_trainable  # noqa: E402
from vlmalign.logprobs import forward_logits, token_logprobs  # noqa: E402
from vlmalign.model import load_model, load_processor  # noqa: E402
from vlmalign.preference import build_pairs  # noqa: E402


def evaluate(model, proc, eval_pairs: dict, items) -> dict:
    out = {}
    for name, pairs in eval_pairs.items():
        out[name] = {"image": preference_accuracy(model, proc, pairs),
                     "blind": preference_accuracy(model, proc, pairs, blind=True)}
    out["vqa"] = vqa_accuracy(model, proc, items)
    return out


@torch.no_grad()
def token_table(model, proc, pairs) -> list[dict]:
    """Per-token log-probs of chosen and rejected for a few pairs, so we can read them."""
    model.eval()
    b = pair_batch(proc, pairs).to(model.device)
    logp, mask = token_logprobs(forward_logits(model, b), b.labels)
    rows = []
    for i, p in enumerate(pairs):
        entry = {"kind": p.kind, "prompt": p.prompt}
        for j, side in ((2 * i, "chosen"), (2 * i + 1, "rejected")):
            ids = b.input_ids[j, 1:][mask[j]].tolist()
            entry[side] = {"tokens": [proc.tokenizer.decode([t]) for t in ids],
                           "logp": [round(v, 3) for v in logp[j][mask[j]].tolist()]}
        rows.append(entry)
    return rows


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--negatives", choices=["balanced", "biased"], default="balanced")
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--pairs", type=int, default=4, help="pairs per step")
    ap.add_argument("--train-pairs", type=int, default=600)
    ap.add_argument("--eval-pairs", type=int, default=150)
    ap.add_argument("--vqa-per-kind", type=int, default=30)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--freeze", default="llm")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default=None)
    args = ap.parse_args(argv)

    set_threads()
    seed_everything(args.seed)
    device = pick_device()
    name = args.name or f"dpo_{args.negatives}"
    proc, model = load_processor(), load_model(device=device)

    train = build_pairs(args.train_pairs, args.negatives, seed=args.seed + 31)
    eval_pairs = {"balanced": build_pairs(args.eval_pairs, "balanced", seed=41),
                  "biased": build_pairs(args.eval_pairs, "biased", seed=42)}
    items = vqa_items(args.vqa_per_kind, 43)

    # The reference is the model before any adapter exists. Score it once and drop it.
    t0 = time.time()
    ref_c, ref_r = reference_logprobs(model, proc, train, device=device)
    print(f"[{name}] reference scored for {len(train)} pairs in {time.time() - t0:.0f}s")

    counts = configure_trainable(model, args.freeze)
    print(f"[{name}] device={device} trainable={counts}")
    before = evaluate(model, proc, eval_pairs, items)
    print("before", json.dumps(before))
    worked = eval_pairs["balanced"][:5]  # round-robin, so one pair of each kind
    tokens_before = token_table(model, proc, worked)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    order = torch.randperm(len(train))
    log, t0 = [], time.time()
    model.train()
    for step in range(args.steps):
        idx = order[(torch.arange(args.pairs) + step * args.pairs) % len(train)]
        batch = pair_batch(proc, [train[int(i)] for i in idx]).to(device)
        pi_c, pi_r = pair_logprobs(model, batch)
        out = dpo_loss(pi_c, pi_r, ref_c[idx].to(device), ref_r[idx].to(device), args.beta)
        opt.zero_grad(set_to_none=True)
        out["loss"].backward()
        gnorm = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sync(device)
        log.append({"step": step, "loss": out["loss"].item(), "margin": out["margin"].mean().item(),
                    "accuracy": out["accuracy"].item(), "grad_norm": gnorm.item(), "elapsed": time.time() - t0})
        if step % 10 == 0 or step == args.steps - 1:
            print(f"step {step:3d} loss={log[-1]['loss']:.4f} margin={log[-1]['margin']:+.3f} "
                  f"pair_acc={log[-1]['accuracy']:.2f} grad={log[-1]['grad_norm']:.2f} t={log[-1]['elapsed']:.0f}s")

    train_seconds = time.time() - t0
    after = evaluate(model, proc, eval_pairs, items)
    print("after", json.dumps(after))
    result = {"args": vars(args), "trainable": counts, "before": before, "after": after, "log": log,
              "train_seconds": train_seconds, "tokens_before": tokens_before,
              "tokens_after": token_table(model, proc, worked)}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / f"{name}.json").write_text(json.dumps(result, indent=1))
    return result


if __name__ == "__main__":
    main()
