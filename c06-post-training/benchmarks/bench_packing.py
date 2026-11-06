"""Padded vs packed SFT batches: how many slots are real tokens, and target tokens per second.

    python benchmarks/bench_packing.py --steps 12
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from posttrain.chat import IGNORE_INDEX, load_tokenizer  # noqa: E402
from posttrain.models import load_policy  # noqa: E402
from posttrain.packing import padding_stats  # noqa: E402
from posttrain.sft import batches, build_examples, masked_lm_loss  # noqa: E402
from posttrain.tasks import split_problems  # noqa: E402
from posttrain.utils import BASE_MODEL, hardware_label, pick_device, polite_threads, save_json, synchronize  # noqa: E402


def speed(model, stream, device, steps: int) -> dict[str, float]:
    """Forward + backward only (no optimizer), after two warm-up steps that compile MPS kernels."""
    done = targets = slots = 0
    t0 = time.time()
    for i in range(steps + 2):
        batch = {k: v.to(device) for k, v in next(stream).items()}
        inputs = {k: batch[k] for k in ("input_ids", "attention_mask", "position_ids") if k in batch}
        masked_lm_loss(model(**inputs, use_cache=False).logits, batch["labels"]).backward()
        model.zero_grad(set_to_none=True)
        synchronize(device)
        if i == 1:
            t0 = time.time()
        elif i > 1:
            done += 1
            targets += int((batch["labels"] != IGNORE_INDEX).sum())
            slots += batch["input_ids"].numel()
    sec = time.time() - t0
    return {"sec_per_step": sec / done, "target_tokens_per_step": targets / done, "slots_per_step": slots / done,
            "target_tokens_per_sec": targets / sec}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--out", default="bench_packing.json")
    args = ap.parse_args()
    polite_threads()
    device = pick_device()
    tok = load_tokenizer()
    train, _ = split_problems(2000, 10, seed=0, follow_up_frac=0.3)
    examples = build_examples(tok, train)
    lengths = [len(e.input_ids) for e in examples]
    stats = padding_stats(lengths, batch_size=8, max_len=256)
    print(f"{len(lengths)} conversations, {min(lengths)}-{max(lengths)} tokens, {stats['real_tokens']} real tokens")
    print(f"padding (8 per batch): {stats['padded_slots']} slots, {stats['pad_efficiency']:.1%} real")
    print(f"packing (256 per row): {stats['packed_slots']} slots, {stats['pack_efficiency']:.1%} real")

    model = load_policy(BASE_MODEL, device).train()
    padded = speed(model, batches(examples, 8, tok.pad_token_id, False, 256, 0), device, args.steps)
    packed = speed(model, batches(examples, 4, tok.pad_token_id, True, 256, 0), device, args.steps)
    print(f"hardware: {hardware_label(device)}")
    print(f"{'batching':<20}{'sec/step':>10}{'targets/step':>14}{'slots/step':>12}{'targets/sec':>13}")
    for name, s in (("padded 8 x bucket", padded), ("packed 4 x 256", packed)):
        print(f"{name:<20}{s['sec_per_step']:>10.2f}{s['target_tokens_per_step']:>14.0f}"
              f"{s['slots_per_step']:>12.0f}{s['target_tokens_per_sec']:>13.0f}")
    save_json({"hardware": hardware_label(device), "stats": stats, "padded": padded, "packed": packed}, args.out)


if __name__ == "__main__":
    main()
