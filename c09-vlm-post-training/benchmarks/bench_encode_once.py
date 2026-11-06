"""Scoring G rollouts of one image: run the vision encoder G times, or once?

    python benchmarks/bench_encode_once.py --groups 4 8 16

Prints wall-clock time for one forward+backward, on the device pick_device() returns.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vlmalign.device import pick_device, set_threads, sync  # noqa: E402
from vlmalign.freeze import configure_trainable  # noqa: E402
from vlmalign.logprobs import build_batch, forward_logits, token_logprobs  # noqa: E402
from vlmalign.model import encode_answer, encode_prompt, load_model, load_processor  # noqa: E402
from vlmalign.scenes import render, sample_scene  # noqa: E402


def timed(fn, device, repeats: int = 5) -> float:
    fn()  # warm-up
    times = []
    for _ in range(repeats):
        sync(device)
        t0 = time.perf_counter()
        fn()
        sync(device)
        times.append(time.perf_counter() - t0)
    return sorted(times)[len(times) // 2] * 1000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", type=int, nargs="+", default=[4, 8, 16])
    ap.add_argument("--freeze", default="all")
    args = ap.parse_args()
    set_threads()
    device = pick_device()
    proc, model = load_processor(), load_model(device=device)
    print("trainable", configure_trainable(model, args.freeze))
    model.train()
    prompt = encode_prompt(proc, render(sample_scene(0)), "How many circles are there? Answer with a number.")
    answer = encode_answer(proc, "3")
    pad = proc.tokenizer.pad_token_id

    print(f"\ndevice={device}  freeze={args.freeze}  one image, G rollouts, median of 5")
    print(f"{'G':>4s} {'naive (ms)':>12s} {'once (ms)':>12s} {'speed-up':>9s}")
    for g in args.groups:
        batch = build_batch([prompt.input_ids[0]] * g, [answer] * g, [0] * g, prompt.pixel_values, pad).to(device)

        def step(encode_once: bool):
            def run():
                model.zero_grad(set_to_none=True)
                logits = forward_logits(model, batch, encode_once=encode_once)
                logp, _ = token_logprobs(logits, batch.labels)
                logp.sum().backward()
            return run

        naive, once = timed(step(False), device), timed(step(True), device)
        print(f"{g:4d} {naive:12.0f} {once:12.0f} {naive / once:8.2f}x")


if __name__ == "__main__":
    main()
