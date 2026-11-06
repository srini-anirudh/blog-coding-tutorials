"""Stage-1 alignment: train only the projector on synthetic shape captions, then compare captions.

    python c07-vlm/scripts/train_alignment.py --projector mlp --steps 300

Vision encoder and language model stay frozen. Gradients still flow *through* the frozen language model
into the projector, because the loss is on the caption tokens and the image features sit in the context.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import transformers
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tinyvlm import build_tinyvlm, describe_device, pick_device  # noqa: E402
from tinyvlm.data import make_example, score  # noqa: E402

TRAIN_SEED, EVAL_SEED = 0, 1


@torch.no_grad()
def caption_eval(model, processor, n: int, device, batch_size: int = 16) -> tuple[list[str], dict]:
    """Greedy captions for n held-out pictures, scored by parsing count / colour / shape."""
    examples = [make_example(i, seed=EVAL_SEED) for i in range(n)]
    predictions = []
    for start in range(0, n, batch_size):
        chunk = examples[start : start + batch_size]
        prompts = [processor.apply_chat_template(e["messages"][:1], add_generation_prompt=True) for e in chunk]
        batch = processor(prompts, [e["images"] for e in chunk])
        new = model.generate(
            batch["input_ids"].to(device), batch["pixel_values"].to(device), batch["attention_mask"].to(device),
            max_new_tokens=12, do_sample=False, pad_token_id=processor.tokenizer.pad_token_id,
        )
        predictions += [p.strip() for p in processor.batch_decode(new)]
    return predictions, score(predictions, [e["target"] for e in examples])


def lr_at(step: int, steps: int, peak: float, warmup: int) -> float:
    """Linear warmup, then cosine decay to 10% of the peak."""
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, steps - warmup)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--projector", default="mlp", choices=["linear", "mlp"])
    parser.add_argument("--pixel-shuffle", type=int, default=2)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--eval-n", type=int, default=96)
    parser.add_argument("--eval-every", type=int, default=50, help="caption a small held-out set during training")
    parser.add_argument("--bad-labels", action="store_true", help="Break it: take the loss on every token")
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    torch.manual_seed(0)
    transformers.logging.disable_progress_bar()

    device = pick_device(args.device)
    model, processor = build_tinyvlm(projector_type=args.projector, pixel_shuffle_factor=args.pixel_shuffle)
    model.set_trainable(vision=False, projector=True, lm=False)
    model.to(device)
    default_name = f"align-{args.projector}-s{args.pixel_shuffle}" + ("-badlabels" if args.bad_labels else "")
    out = Path(args.out or Path(__file__).resolve().parents[1] / "runs" / default_name)
    out.mkdir(parents=True, exist_ok=True)
    print(f"device {describe_device(device)} | params {model.parameter_counts()}")

    before, before_score = caption_eval(model, processor, args.eval_n, device)
    print("before:", before_score, before[:3], flush=True)

    optimizer = torch.optim.AdamW(model.projector.parameters(), lr=args.lr, weight_decay=0.0)
    history, snapshots, trained_seconds = [], [{"step": 0, "score": before_score, "captions": before[:6]}], 0.0
    model.train()
    for step in range(args.steps):
        start = time.time()
        examples = [make_example(step * args.batch_size + i, seed=TRAIN_SEED) for i in range(args.batch_size)]
        batch = processor.collate([processor.build_example(e["messages"], e["images"]) for e in examples])
        if args.bad_labels:  # every real token becomes a target, image placeholders included
            batch["labels"] = batch["input_ids"].masked_fill(batch["attention_mask"] == 0, -100)
        loss = model(
            input_ids=batch["input_ids"].to(device),
            pixel_values=batch["pixel_values"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            labels=batch["labels"].to(device),
        ).loss
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, args.steps, args.lr, args.warmup)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.projector.parameters(), 1.0)
        optimizer.step()
        trained_seconds += time.time() - start
        history.append({"step": step, "loss": loss.item(), "grad_norm": grad_norm.item()})
        if step % 10 == 0 or step == args.steps - 1:
            print(f"step {step:4d} loss {loss.item():.3f} grad_norm {grad_norm.item():.2f} {trained_seconds:6.1f}s", flush=True)
        if (step + 1) % args.eval_every == 0 and step + 1 < args.steps:
            model.eval()
            captions, snap = caption_eval(model, processor, min(32, args.eval_n), device)
            snapshots.append({"step": step + 1, "score": snap, "captions": captions[:6]})
            print(f"  eval at {step + 1}: {snap} {captions[:2]}", flush=True)
            model.train()
    model.eval()

    after, after_score = caption_eval(model, processor, args.eval_n, device)
    print("after:", after_score, after[:3])
    snapshots.append({"step": args.steps, "score": after_score, "captions": after[:6]})
    save_file(model.projector.state_dict(), out / "projector.safetensors")
    report = {
        "device": describe_device(device), "args": {k: v for k, v in vars(args).items() if k != "out"},
        "params": model.parameter_counts(), "train_seconds": trained_seconds, "before": before_score, "after": after_score,
        "samples_before": before[:12], "samples_after": after[:12], "snapshots": snapshots, "history": history,
    }
    results = Path(__file__).resolve().parents[1] / "results"
    results.mkdir(exist_ok=True)
    (results / f"{out.name}.json").write_text(json.dumps(report, indent=1))  # small; kept in git, weights are not
    print(f"saved {out.name} | {trained_seconds:.0f}s training")


if __name__ == "__main__":
    main()
