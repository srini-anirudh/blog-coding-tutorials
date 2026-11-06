"""Sections 7, 21, 22 and 19: prefix caching with chained block hashes.

Part A (GPU, real model): 64 requests share a 2,048-token system prompt and each adds its own 64-token
question. With the cache on, the shared blocks are computed once. Outputs must be identical either way.
Part B (block manager only): eight requests over four prefixes, a pool that holds about two prefixes.
Arrival order decides the hit rate; grouping requests by prefix (what a radix-tree scheduler can do)
halves the prefill work.
Part C (block manager only): two image prompts with the same placeholder ids. Without the image hash in
the block key, the second request is told the first image's KV is a valid prefix.
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from servekit.block_manager import BlockManager  # noqa: E402
from servekit.engine import Engine  # noqa: E402
from servekit.metrics import Report, device_and_dtype, random_prompts, summarize  # noqa: E402
from servekit.model import load_model  # noqa: E402
from servekit.scheduler import SchedulerConfig  # noqa: E402


def dev_is_cuda() -> bool:
    return device_and_dtype()[0] == "cuda"


def part_a(rep, model_name: str, sys_len: int, n: int, fp32: bool = False) -> list[dict]:
    dev, dtype = device_and_dtype()
    if fp32:
        dtype = torch.float32
    model = load_model(model_name, dtype, dev)
    system = random_prompts(1, sys_len, sys_len, model.config.vocab_size, seed=3)[0]
    prompts = [system + q for q in random_prompts(n, 64, 64, model.config.vocab_size, seed=4)]
    warm = [system + q for q in random_prompts(4, 64, 64, model.config.vocab_size, seed=5)]
    rows, outputs = [], []
    for cache in (False, True):
        eng = Engine(model, num_blocks=4000, enable_prefix_caching=cache,
                     sched=SchedulerConfig(max_num_seqs=16, max_num_batched_tokens=4096))
        eng.run([(0.0, p, 32) for p in warm])  # warm-up; with the cache on it also leaves the system prompt cached
        eng.finished.clear()
        eng.stats.clear()
        eng.sched.prompt_tokens = eng.sched.cached_tokens = 0
        wall = eng.run([(0.0, p, 32) for p in prompts])
        s = summarize(eng.finished, wall)
        outputs.append([q.output for q in sorted(eng.finished, key=lambda q: q.seq_id)])
        computed = sum(st.num_prefill_tokens for st in eng.stats)
        rows.append({"dtype": str(dtype).removeprefix("torch."), "prefix_cache": "on" if cache else "off", "prompt_tokens": eng.sched.prompt_tokens,
                     "cached_tokens": eng.sched.cached_tokens, "prefill_tokens_computed": computed,
                     "wall_s": wall, "ttft_p50_ms": s["ttft_p50"] * 1e3, "ttft_p99_ms": s["ttft_p99"] * 1e3})
        del eng
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    rep(f"Part A ({str(dtype).removeprefix('torch.')}): {model_name}, {n} requests = shared {sys_len}-token system prompt + own 64-token question, 32 output tokens")
    rep.table(rows, {"wall_s": "{:.2f}", "ttft_p50_ms": "{:.1f}", "ttft_p99_ms": "{:.1f}"})
    same = sum(a == b for a, b in zip(*outputs)) / len(outputs[0])
    rep(f"requests whose 32 greedy tokens are identical with and without the cache: {same:.2f}")
    return rows


def prefill_work(order: list[str], prefixes: dict[str, list[int]], pool_blocks: int, bs: int = 16) -> int:
    bm = BlockManager(pool_blocks, bs, enable_prefix_caching=True)
    computed = 0
    for i, name in enumerate(order):
        prompt = prefixes[name] + [10_000 + i] * bs  # each request adds one block of its own
        cached = bm.allocate_prompt(i, prompt)
        assert bm.ensure(i, len(prompt))
        bm.register_computed(i, prompt, len(prompt))
        computed += len(prompt) - cached
        bm.free_seq(i)
    return computed


def part_b(rep) -> dict:
    prefixes = {c: [ord(c) * 100 + j for j in range(64)] for c in "ABCD"}  # 4 blocks each
    pool = 10  # room for two cached prefixes plus one request in flight
    arrival = list("ABCDABCD")
    grouped = sorted(arrival)
    a, g = prefill_work(arrival, prefixes, pool), prefill_work(grouped, prefixes, pool)
    rep("")
    rep(f"Part B: 8 requests over prefixes A-D (64 tokens each) plus 16 own tokens; pool of {pool} blocks of 16")
    rep(f"  arrival order {''.join(arrival)}: {a} prefill tokens computed")
    rep(f"  grouped order {''.join(grouped)}: {g} prefill tokens computed ({g / a:.2f} of arrival order)")
    return {"arrival_tokens": a, "grouped_tokens": g}


def part_c(rep) -> dict:
    placeholder = [32000] * 64  # the same <image> ids for any image
    question = [5, 6, 7, 8] * 4
    out = {}
    for with_hash in (False, True):
        bm = BlockManager(64, 16, enable_prefix_caching=True)
        cat = ["cat-image-hash"] * 5 if with_hash else None
        dog = ["dog-image-hash"] * 5 if with_hash else None
        p = placeholder + question + [1]
        bm.allocate_prompt(0, p, cat)
        bm.ensure(0, len(p))
        bm.register_computed(0, p, len(p), cat)
        out["with_image_hash" if with_hash else "token_ids_only"] = bm.allocate_prompt(1, p, dog)
    rep("")
    rep("Part C: cat image, then dog image, same question. Tokens the dog request is told are cached:")
    rep(f"  hash of token ids only: {out['token_ids_only']} (wrong: that is the cat's KV)")
    rep(f"  image hash in the key:  {out['with_image_hash']}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-360M")
    ap.add_argument("--system-len", type=int, default=2048)
    ap.add_argument("--requests", type=int, default=64)
    ap.add_argument("--skip-model", action="store_true")
    args = ap.parse_args()
    rep = Report("04_prefix_cache.txt")
    a = [] if args.skip_model else part_a(rep, args.model, args.system_len, args.requests)
    if not args.skip_model and dev_is_cuda():
        rep("")
        a += part_a(rep, args.model, args.system_len, args.requests, fp32=True)
    b, c = part_b(rep), part_c(rep)
    rep.save({"part_a": a, "part_b": b, "part_c": c})


if __name__ == "__main__":
    main()
