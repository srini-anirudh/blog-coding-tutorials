"""Sections 4 and 21: where num_gpu_blocks comes from, and recompute versus swap.

Part A carves the GPU's memory down to a block count the way vLLM's startup does: measure the weights after
loading, run one profiling forward pass at the largest batch the scheduler allows, and give what is left
(up to gpu_memory_utilization) to KV blocks.
Part B times both ways of bringing back a preempted 8,000-token sequence: recompute (prefill it again) and
swap (copy its KV blocks to pinned host memory and back over PCIe).
Part C runs a workload through a deliberately small pool and counts preemptions.

Default model: llama3-8b-random, Llama-3-8B shapes with random weights (speed and memory do not depend on the
weight values; the text it produces is meaningless). Pass --model HuggingFaceTB/SmolLM2-360M for a real checkpoint.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from servekit.engine import Engine  # noqa: E402
from servekit.metrics import Report, output_lengths, random_prompts, summarize  # noqa: E402
from servekit.model import BatchMeta, PagedKVCache, kv_bytes_per_token, load_model  # noqa: E402
from servekit.scheduler import SchedulerConfig  # noqa: E402

BS = 16


def part_a(rep, model, max_tokens: int) -> dict:
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.synchronize()
    weights = torch.cuda.memory_allocated()
    # Profiling run: max_tokens tokens of prefill through a throwaway one-sequence cache.
    cache = PagedKVCache(model.config, max_tokens // BS, BS, torch.bfloat16, "cuda")
    cache_bytes = cache.nbytes()
    torch.cuda.reset_peak_memory_stats()
    dev = "cuda"
    meta = BatchMeta(torch.randint(0, 1000, (max_tokens,), device=dev), torch.arange(max_tokens, device=dev),
                     torch.arange(max_tokens, device=dev), torch.tensor([max_tokens - 1], device=dev),
                     prefills=[(0, max_tokens, max_tokens, torch.arange(max_tokens, device=dev), None)])
    model(meta, cache)
    torch.cuda.synchronize()
    activation_peak = torch.cuda.max_memory_allocated() - weights - cache_bytes
    del cache, meta
    per_block = kv_bytes_per_token(model.config, 2) * BS
    out = {"total_gib": total / 2**30, "weights_gib": weights / 2**30, "activation_peak_gib": activation_peak / 2**30,
           "bytes_per_block": per_block}
    rep(f"Part A: {torch.cuda.get_device_name(0)}, total {out['total_gib']:.1f} GiB")
    rep(f"  weights after loading (bf16)            {out['weights_gib']:.3f} GiB")
    rep(f"  activation peak, {max_tokens}-token profiling pass {out['activation_peak_gib']:.3f} GiB")
    rep(f"  KV per block = {kv_bytes_per_token(model.config, 2)} B/token x {BS} = {per_block / 2**10:.0f} KiB")
    for util in (0.85, 0.90, 0.95):
        kv = total * util - weights - activation_peak
        blocks = int(kv // per_block)
        out[f"blocks_at_{util}"] = blocks
        rep(f"  gpu_memory_utilization {util}: {kv / 2**30:.1f} GiB for KV = {blocks:,} blocks = "
            f"{blocks * BS:,} tokens = {blocks * BS // 4096:,} sequences of 4k")
    return out


def part_b(rep, model, n: int) -> dict:
    eng = Engine(model, num_blocks=n // BS + 8, sched=SchedulerConfig(max_num_seqs=1, max_num_batched_tokens=n + 16),
                 max_model_len=n + 16)
    prompt = random_prompts(1, n, n, model.config.vocab_size, seed=9)[0]
    times = []
    for _ in range(3):  # recompute = one full prefill of the sequence
        eng.run([(0.0, prompt, 1)])
        times.append(eng.stats[-1].seconds)
    recompute = sorted(times)[1]
    blocks = torch.arange(n // BS, device="cuda")
    k, v = eng.cache.k[:, blocks], eng.cache.v[:, blocks]
    host_k = torch.empty(k.shape, dtype=k.dtype, pin_memory=True)
    host_v = torch.empty(v.shape, dtype=v.dtype, pin_memory=True)
    host_k.zero_(), host_v.zero_()  # touch every page once
    swaps = []
    for _ in range(12):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        host_k.copy_(k, non_blocking=True)
        host_v.copy_(v, non_blocking=True)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        eng.cache.k[:, blocks] = host_k.to("cuda", non_blocking=True)
        eng.cache.v[:, blocks] = host_v.to("cuda", non_blocking=True)
        torch.cuda.synchronize()
        swaps.append((t1 - t0, time.perf_counter() - t1))
    swaps = swaps[2:]  # the first two round trips are warm-up
    out_s, in_s = sorted(swaps)[len(swaps) // 2]
    best_out, best_in = min(x[0] for x in swaps), min(x[1] for x in swaps)
    nbytes = 2 * k.numel() * k.element_size()
    res = {"tokens": n, "kv_mib": nbytes / 2**20, "recompute_ms": recompute * 1e3, "swap_out_ms": out_s * 1e3,
           "swap_in_ms": in_s * 1e3, "swap_round_trip_ms": (out_s + in_s) * 1e3,
           "pcie_gb_s_out": nbytes / out_s / 1e9, "pcie_gb_s_in": nbytes / in_s / 1e9}
    rep("")
    rep(f"Part B: one {n}-token sequence, {res['kv_mib']:.0f} MiB of KV")
    rep(f"  recompute (prefill again)  {res['recompute_ms']:.1f} ms")
    rep(f"  swap out + in over PCIe    {res['swap_out_ms']:.1f} + {res['swap_in_ms']:.1f} = {res['swap_round_trip_ms']:.1f} ms"
        f"  ({res['pcie_gb_s_out']:.1f} / {res['pcie_gb_s_in']:.1f} GB/s)")
    rep(f"  best of 10: out {best_out * 1e3:.1f} ms ({nbytes / best_out / 1e9:.1f} GB/s), in {best_in * 1e3:.1f} ms ({nbytes / best_in / 1e9:.1f} GB/s)")
    rep(f"  recompute / swap = {res['recompute_ms'] / res['swap_round_trip_ms']:.2f}  (swap loses when this is below 1)")
    rep(f"  the post assumes 50 GB/s each way: swap would be {2 * nbytes / 50e9 * 1e3:.1f} ms round trip,"
        f" recompute / swap = {res['recompute_ms'] / (2 * nbytes / 50e9 * 1e3):.1f}")
    res.update(best_out_ms=best_out * 1e3, best_in_ms=best_in * 1e3)
    return res


def part_c(rep, model) -> list[dict]:
    v = model.config.vocab_size
    prompts = random_prompts(48, 64, 256, v, seed=11)
    outs = output_lengths(48, 64, 512, seed=11)
    rows = []
    for blocks in (4000, 600, 300):
        eng = Engine(model, num_blocks=blocks, sched=SchedulerConfig(max_num_seqs=48, max_num_batched_tokens=2048))
        wall = eng.run([(0.0, p, n) for p, n in zip(prompts, outs)])
        s = summarize(eng.finished, wall)
        rows.append({"kv_blocks": blocks, "preemptions": eng.sched.num_preemptions, "wall_s": wall,
                     "out_tok/s": s["out_tok_per_s"], "prefill_tokens": sum(st.num_prefill_tokens for st in eng.stats)})
    rep("")
    rep("Part C: 48 requests (prompts 64-256, outputs 64-512), shrinking the KV pool; preemption is by recompute")
    rep.table(rows, {"wall_s": "{:.2f}", "out_tok/s": "{:.0f}"})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3-8b-random")
    ap.add_argument("--tokens", type=int, default=8000)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        print("needs a CUDA GPU")
        return
    model = load_model(args.model, torch.bfloat16, "cuda")
    rep = Report("06_blocks_and_preemption.txt")
    a = part_a(rep, model, 8192)
    b = part_b(rep, model, args.tokens)
    c = part_c(rep, model)
    rep.save({"num_gpu_blocks": a, "recompute_vs_swap": b, "preemption": c})


if __name__ == "__main__":
    main()
