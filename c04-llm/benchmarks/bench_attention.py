"""Naive attention vs the fused SDPA kernel, inside the full model: time for forward + backward.

On CUDA it also reports peak allocated memory. FlashAttention kernels only accept bf16 or half,
so a float32 run on a GPU does not reach them: pass --dtype bf16 to measure them.

    python c04-llm/benchmarks/bench_attention.py --device cpu
    python c04-llm/benchmarks/bench_attention.py --device cuda --dtype bf16
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tinyllm import TinyLLM, get_config  # noqa: E402
from tinyllm.utils import Timer, device_label, pick_device  # noqa: E402


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
    p.add_argument("--dtype", default="float32", choices=["float32", "bf16"])
    args = p.parse_args()

    device = pick_device(args.device)
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    is_cuda = device.type == "cuda"
    dtype_note = "" if args.dtype == "float32" else f" dtype={args.dtype}"
    print(f"device={device_label(device)}{dtype_note} batch={args.batch_size}  "
          "(tiny preset, forward + backward per step)")
    memory_header = f" {'naive MiB':>10} {'sdpa MiB':>10}" if is_cuda else ""
    print(f"{'seq':>6} {'naive ms':>10} {'sdpa ms':>10} {'speed-up':>9}{memory_header}")
    for seq in [int(s) for s in args.seq_lens.split(",")]:
        times, peaks = {}, {}
        for backend in ("naive", "sdpa"):
            torch.manual_seed(0)
            model = TinyLLM(get_config("tiny", max_seq_len=seq, attn_backend=backend)).to(device, dtype=dtype).train()
            ids = torch.randint(0, 256, (args.batch_size, seq), device=device)
            if is_cuda:
                torch.cuda.reset_peak_memory_stats(device)
            times[backend] = time_train_step(model, ids, args.repeats)
            if is_cuda:
                peaks[backend] = torch.cuda.max_memory_allocated(device) / 2**20
        memory = f" {peaks['naive']:>10.0f} {peaks['sdpa']:>10.0f}" if is_cuda else ""
        print(f"{seq:>6} {1e3 * times['naive']:>10.1f} {1e3 * times['sdpa']:>10.1f} "
              f"{times['naive'] / times['sdpa']:>8.2f}x{memory}", flush=True)


if __name__ == "__main__":
    main()
