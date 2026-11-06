"""Post sections: Training cost vs inference cost ("measure, do not compute"), The quadratic wall (memory).

Part 1 times one block of each kind at Swin's stage-1 shape (56 x 56 tokens, D = 96, 3 heads) and divides the
measured FLOPs by the time, so blocks with very different FLOP counts can be compared on achieved GFLOP/s.
Part 2 (CUDA only) measures peak memory of one ViT-B attention layer, forward and backward, with the score matrix
materialised (naive) and with fused SDPA, from 224 to 1024 px.

    python labs/an-image-is-a-sentence/scripts/04_wallclock.py --device cpu
    python labs/an-image-is-a-sentence/scripts/04_wallclock.py --device cuda
Saves results/04_wallclock_<device>.txt and .json.
"""

import argparse
import statistics
import time

import _setup  # noqa: F401
import torch

from vitfamily.attention import MHSA
from vitfamily.blocks import PVTBlock, SwinBlock, ViTBlock
from vitfamily.c03 import be_polite
from vitfamily.flops import measure_flops
from vitfamily.report import Report

p = argparse.ArgumentParser()
p.add_argument("--device", default="cpu")
p.add_argument("--batch", type=int, default=None, help="default: 4 on CPU, 64 on CUDA")
p.add_argument("--iters", type=int, default=10)
args = p.parse_args()
be_polite(8)
dev = torch.device(args.device)
cuda = dev.type == "cuda"
batch = args.batch or (64 if cuda else 4)
dtype = torch.bfloat16 if cuda else torch.float32
log = Report(f"04_wallclock_{dev.type}", device=dev.type)


def sync() -> None:
    if cuda:
        torch.cuda.synchronize()


def timed(fn) -> float:
    for _ in range(3):
        fn()
    sync()
    ts = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        fn()
        sync()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


log(f"== Part 1: one block forward, batch {batch}, {dtype}, inference mode, median of {args.iters} ==")
grid = (1, 56, 56, 96)
cases = {
    "ViT global, 3136 tokens": (lambda: ViTBlock(96, 3), (1, 3136, 96), ()),
    "Swin W-MSA, M=7": (lambda: SwinBlock(96, 3, 7, 0), grid, ()),
    "Swin SW-MSA, M=7, shift 3": (lambda: SwinBlock(96, 3, 7, 3), grid, ()),
    "PVT SRA, R=8": (lambda: PVTBlock(96, 3, 8), (1, 3136, 96), ((56, 56),)),
    "ViT global, 196 tokens, D=384": (lambda: ViTBlock(384, 6), (1, 196, 384), ()),
}
log(f"{'block':<32}{'GFLOPs/sample':>14}{'ms':>10}{'GFLOP/s':>10}")
part1 = {}
for name, (build, shape, extra) in cases.items():
    with torch.device("meta"):
        fl = measure_flops(lambda: build()(torch.empty(*shape), *extra))["total"]
    torch.manual_seed(0)
    blk = build().to(dev, dtype).eval()
    x = torch.randn(batch, *shape[1:], device=dev, dtype=dtype)
    with torch.inference_mode():
        sec = timed(lambda: blk(x, *extra))
    rate = fl * batch / sec / 1e9
    part1[name] = {"gflops_per_sample": fl / 1e9, "ms": sec * 1e3, "gflop_per_s": rate}
    log(f"{name:<32}{fl / 1e9:>14.3f}{sec * 1e3:>10.2f}{rate:>10.0f}")
log.data["blocks"] = part1

if cuda:
    log("\n== Part 2: one ViT-B attention layer (12 heads, D=768), forward + backward, batch 1, bf16 ==")
    log(f"{'image':>6}{'N':>6}{'naive MiB':>12}{'SDPA MiB':>11}{'naive ms':>10}{'SDPA ms':>10}{'scores, MiB':>13}")
    part2 = {}
    attn = MHSA(768, 12).to(dev, dtype)
    for px in (224, 512, 768, 1024):
        n = (px // 16) ** 2 + 1
        row = {}
        for mode in ("naive", "sdpa"):
            x = torch.randn(1, n, 768, device=dev, dtype=dtype, requires_grad=True)

            def step():
                out = attn(x, return_attn=True)[0] if mode == "naive" else attn(x)
                out.float().sum().backward()

            step()
            sync()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            step()
            sync()
            row[f"{mode}_mib"] = (torch.cuda.max_memory_allocated() - base) / 2**20
            row[f"{mode}_ms"] = timed(step) * 1e3
        row["scores_mib"] = 12 * n * n * 2 / 2**20
        part2[px] = row
        log(f"{px:>6}{n:>6}{row['naive_mib']:>12.0f}{row['sdpa_mib']:>11.0f}{row['naive_ms']:>10.2f}{row['sdpa_ms']:>10.2f}"
            f"{row['scores_mib']:>13.0f}")
    log.data["attention_memory"] = part2
log.save()
