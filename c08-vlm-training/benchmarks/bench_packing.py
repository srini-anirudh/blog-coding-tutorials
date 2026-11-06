"""Padding vs length grouping vs packing on one real mixture of samples.

    python c08-vlm-training/benchmarks/bench_packing.py --n 480 --budget 512

The first table only counts tokens. The second one times real forward and backward passes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vlmtrain.c07_adapter import build_c07  # noqa: E402
from vlmtrain.data import TASK_SIZES, make_example  # noqa: E402
from vlmtrain.device import laptop_threads, memory_mb, pick_device, sync  # noqa: E402
from vlmtrain.freeze import set_trainable  # noqa: E402
from vlmtrain.mixture import MixtureSampler, mixture_weights  # noqa: E402
from vlmtrain.packing import first_fit_decreasing, pack, padding_fraction  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=480)
    ap.add_argument("--budget", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--timed", type=int, default=64, help="samples used for the timed comparison (0 skips it)")
    args = ap.parse_args()
    laptop_threads(2)

    device = pick_device()
    tok, model, proc = build_c07() if args.timed else (None, None, None)
    if proc is None:
        from vlmtrain.processor import TinyVLMProcessor
        from vlmtrain.tokenizer import build_tokenizer

        proc = TinyVLMProcessor(build_tokenizer())
    sampler = MixtureSampler(mixture_weights(TASK_SIZES, "temperature", 2.0), TASK_SIZES, seed=0)
    enc = []
    for _ in range(args.n):
        t, i = sampler.draw()
        enc.append(proc.encode(make_example(t, i)))
    lens = [len(e.input_ids) for e in enc]
    total = sum(lens)
    rows = first_fit_decreasing(lens, args.budget)
    pad_random = padding_fraction(lens, args.batch_size)
    pad_sorted = padding_fraction(lens, args.batch_size, sort=True)
    pack_waste = 1 - total / (len(rows) * args.budget)
    print(f"{args.n} samples, {total:,} real tokens, length min {min(lens)} / mean {total / len(lens):.0f} / max {max(lens)}")
    print(f"{'layout':<26}{'rows':>6}{'positions':>11}{'wasted':>9}")
    def line(name, n_rows, positions, waste):
        print(f"{name:<26}{n_rows:>6}{positions:>11,}{waste:>9.1%}")
    chunks = [lens[i:i + args.batch_size] for i in range(0, len(lens), args.batch_size)]
    line(f"pad, random (B={args.batch_size})", args.n, sum(max(c) * len(c) for c in chunks), pad_random)
    slens = sorted(lens)
    schunks = [slens[i:i + args.batch_size] for i in range(0, len(slens), args.batch_size)]
    line(f"pad, grouped (B={args.batch_size})", args.n, sum(max(c) * len(c) for c in schunks), pad_sorted)
    line(f"pack (budget {args.budget})", len(rows), len(rows) * args.budget, pack_waste)
    res = {"n": args.n, "budget": args.budget, "real_tokens": total, "pad_random": pad_random, "pad_grouped": pad_sorted,
           "pack_waste": pack_waste, "rows": len(rows), "min": min(lens), "max": max(lens), "mean": total / len(lens)}

    if args.timed:
        model.to(device).train()
        set_trainable(model, "lm")
        sub = enc[: args.timed]
        real = sum(len(e.input_ids) for e in sub)

        def step(fwd) -> None:
            model.zero_grad(set_to_none=True)
            fwd().sum().backward()

        def run_pad() -> float:
            positions = 0
            t0 = time.time()
            for i in range(0, len(sub), args.batch_size):
                b = proc.collate(sub[i:i + args.batch_size])
                positions += b["input_ids"].numel()
                step(lambda: model(b["input_ids"].to(device), b["pixel_values"].to(device),
                                   attention_mask=b["attention_mask"].to(device)).logits[:, -1])
            sync(device)
            return time.time() - t0

        def run_pack() -> float:
            b = pack(sub, args.budget, proc.pad_id)
            t0 = time.time()
            for r in range(b["input_ids"].shape[0]):
                # one row at a time, so peak memory stays comparable; tiles are selected per row
                offs = sum(sum(sub[i].pixel_values.shape[0] for i in row) for row in b["rows"][:r])
                nt = sum(sub[i].pixel_values.shape[0] for i in b["rows"][r])
                step(lambda: model(b["input_ids"][r:r + 1].to(device), b["pixel_values"][offs:offs + nt].to(device),
                                   attention_mask=b["attention_mask"][r:r + 1].to(device),
                                   position_ids=b["position_ids"][r:r + 1].to(device)).logits[:, -1])
            sync(device)
            return time.time() - t0

        print(f"\nTimed forward + backward on {device}, {args.timed} samples, {real:,} real tokens")
        print(f"{'layout':<26}{'seconds':>9}{'real tok/s':>12}")
        tp = run_pad()
        print(f"{'pad, random':<26}{tp:>9.1f}{real / tp:>12,.0f}")
        tk = run_pack()
        print(f"{'pack':<26}{tk:>9.1f}{real / tk:>12,.0f}")
        res.update({"timed": {"device": str(device), "samples": args.timed, "real": real, "pad_sec": tp, "pack_sec": tk,
                              "mem_mb": memory_mb(device)}})
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "bench_packing.json").write_text(json.dumps(res, indent=1))
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


if __name__ == "__main__":
    main()
