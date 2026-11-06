"""Naive attention vs the fused SDPA kernel, inside the full model: forward + backward time and memory.

    python c04-llm/benchmarks/bench_attention.py --device mps
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tinyllm import TinyLLM, get_config  # noqa: E402
from tinyllm.utils import Timer, pick_device  # noqa: E402


def time_train_step(model: TinyLLM, ids: torch.Tensor, repeats: int) -> float:
    for _ in range(2):  # warm-up
        model(ids, labels=ids).loss.backward()
    with Timer(ids.device) as t:
        for _ in range(repeats):
            model(ids, labels=ids).loss.backward()
    return t.seconds / repeats


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default=None)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seq-lens", default="128,256,512,1024")
    p.add_argument("--repeats", type=int, default=5)
    args = p.parse_args()

    torch.set_num_threads(2)
    device = pick_device(args.device)
    print(f"device={device} batch={args.batch_size}  (tiny preset, forward + backward per step)")
    print(f"{'seq':>6} {'naive ms':>10} {'sdpa ms':>10} {'speed-up':>9}")
    for seq in [int(s) for s in args.seq_lens.split(",")]:
        times = {}
        for backend in ("naive", "sdpa"):
            torch.manual_seed(0)
            model = TinyLLM(get_config("tiny", max_seq_len=seq, attn_backend=backend)).to(device).train()
            ids = torch.randint(0, 256, (args.batch_size, seq), device=device)
            times[backend] = time_train_step(model, ids, args.repeats)
        print(f"{seq:>6} {1e3 * times['naive']:>10.1f} {1e3 * times['sdpa']:>10.1f} {times['naive'] / times['sdpa']:>8.2f}x")


if __name__ == "__main__":
    main()
