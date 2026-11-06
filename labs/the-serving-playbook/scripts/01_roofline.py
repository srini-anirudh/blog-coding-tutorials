"""Section 1: the batch-size free lunch. A decode step reads every weight once, whatever the batch size.

Part A times one Llama-3-8B decoder layer's GEMMs (random bf16 weights: q, k, v, o, gate, up, down) for a
batch of B token rows. Bytes moved stay near the weight size, FLOPs grow with B, so time stays flat until B
reaches the balance point and then grows linearly. Part B times a whole SmolLM2-360M decode step through
servekit's paged forward pass, which is far too small to be bandwidth-bound: it is launch-bound (Section 15).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from servekit.metrics import Report  # noqa: E402

D, F, HKV = 4096, 14336, 8 * 128  # Llama-3-8B hidden size, MLP width, KV width


def time_cuda(fn, iters: int = 50) -> float:
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(iters):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / iters / 1e3


def layer_gemms(batches: list[int], rep) -> list[dict]:
    g = torch.Generator(device="cuda").manual_seed(0)
    shapes = [(D, D), (D, HKV), (D, HKV), (D, D), (D, F), (D, F), (F, D)]
    ws = [torch.randn(i, o, device="cuda", dtype=torch.bfloat16, generator=g) * 0.02 for i, o in shapes]
    wbytes = sum(w.numel() * 2 for w in ws)
    params = sum(w.numel() for w in ws)
    rows = []
    for b in batches:
        xs = {w.shape[0]: torch.randn(b, w.shape[0], device="cuda", dtype=torch.bfloat16) for w in ws}

        def run():
            for w in ws:
                xs[w.shape[0]] @ w

        t = time_cuda(run)
        flops = 2 * params * b
        nbytes = wbytes + sum(2 * b * (w.shape[0] + w.shape[1]) for w in ws)
        rows.append({"B": b, "us": t * 1e6, "TFLOP/s": flops / t / 1e12, "GB/s": nbytes / t / 1e9,
                     "FLOP/byte": flops / nbytes})
    if hasattr(rep, "table"):
        rep(f"Part A: one Llama-3-8B layer's GEMMs, {params / 1e6:.0f} M parameters, {wbytes / 2**20:.0f} MiB of bf16 weights")
        rep.table(rows, {"us": "{:.1f}", "TFLOP/s": "{:.1f}", "GB/s": "{:.0f}", "FLOP/byte": "{:.1f}"})
    return rows


def model_decode(batches: list[int], ctx: int, rep) -> list[dict]:
    from servekit.engine import Engine
    from servekit.metrics import random_prompts
    from servekit.model import load_model
    from servekit.scheduler import SchedulerConfig

    model = load_model("HuggingFaceTB/SmolLM2-360M", torch.bfloat16, "cuda")
    rows = []
    for b in batches:
        eng = Engine(model, num_blocks=b * (ctx // 16 + 4), sched=SchedulerConfig(max_num_seqs=b, max_num_batched_tokens=b * ctx))
        for p in random_prompts(b, ctx, ctx, model.config.vocab_size, seed=b):
            eng.add_request(p, 40)
        eng.step()  # prefill
        for _ in range(39):
            eng.step()
        dec = [s.seconds for s in eng.stats[1:]][5:]
        t = sorted(dec)[len(dec) // 2]
        rows.append({"B": b, "ms_per_step": t * 1e3, "tok/s": b / t})
    rep(f"Part B: SmolLM2-360M decode step, eager PyTorch, context {ctx}, median of 34 steps")
    rep.table(rows, {"ms_per_step": "{:.2f}", "tok/s": "{:.0f}"})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-model", action="store_true")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        print("needs a CUDA GPU")
        return
    rep = Report("01_roofline.txt")
    a = layer_gemms([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096], rep)
    rep("")
    rep("Section 14: speculative decoding with gamma = 4 verifies 5 positions per sequence.")
    spec = layer_gemms([b for b0 in (1, 4, 16, 64, 256) for b in (b0, 5 * b0)], lambda *x: None)
    t = {r["B"]: r["us"] for r in spec}
    spec_rows = [{"B": b, "us_B_rows": t[b], "us_5B_rows": t[5 * b], "cost_ratio": t[5 * b] / t[b]} for b in (1, 4, 16, 64, 256)]
    rep.table(spec_rows, {"us_B_rows": "{:.1f}", "us_5B_rows": "{:.1f}", "cost_ratio": "{:.2f}"})
    rep("")
    b_rows = [] if args.skip_model else model_decode([1, 4, 16, 64, 128], 256, rep)
    rep.save({"layer_gemms": a, "spec_verify": spec_rows, "smollm2_decode": b_rows})


if __name__ == "__main__":
    main()
