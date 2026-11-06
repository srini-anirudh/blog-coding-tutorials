"""Generation speed with and without the KV cache.

Without a cache, step t re-runs the whole t-token sequence, so per-token latency grows with t.
With a cache, step t runs one token against t cached keys, so latency stays nearly flat.

    python c04-llm/benchmarks/bench_generation.py --device mps
    python c04-llm/benchmarks/bench_generation.py --device cpu --preset smollm2-135m --new-tokens 128
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tinyllm import KVCache, TinyLLM, get_config  # noqa: E402
from tinyllm.utils import pick_device, synchronize  # noqa: E402


@torch.no_grad()
def step_latencies(model: TinyLLM, prompt: torch.Tensor, new_tokens: int, use_cache: bool) -> list[float]:
    """Greedy decode; return the wall time of every step in seconds (step 0 includes the prefill)."""
    device = prompt.device
    ids, cache, times = prompt, None, []
    for _ in range(new_tokens):
        synchronize(device)
        t0 = time.perf_counter()
        if not use_cache:
            logits = model(ids).logits
        elif cache is None:
            cache = KVCache()
            logits = model(ids, past_key_values=cache).logits
        else:
            logits = model(ids[:, -1:], past_key_values=cache).logits
        next_token = logits[:, -1].argmax(-1, keepdim=True)
        ids = torch.cat((ids, next_token), dim=1)
        synchronize(device)
        times.append(time.perf_counter() - t0)
    return times


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--preset", default="tiny")
    p.add_argument("--device", default=None)
    p.add_argument("--prompt-len", type=int, default=16)
    p.add_argument("--new-tokens", type=int, default=496)
    p.add_argument("--batch-size", type=int, default=1)
    args = p.parse_args()

    torch.set_num_threads(2)
    torch.manual_seed(0)
    device = pick_device(args.device)
    config = get_config(args.preset, max_seq_len=args.prompt_len + args.new_tokens)
    model = TinyLLM(config).to(device).eval()  # random weights: speed does not depend on them
    prompt = torch.randint(0, config.vocab_size, (args.batch_size, args.prompt_len), device=device)

    step_latencies(model, prompt, 8, True)  # warm-up: kernel compilation, allocator
    step_latencies(model, prompt, 8, False)

    print(f"preset={args.preset} params={model.num_parameters():,} device={device} batch={args.batch_size} "
          f"prompt={args.prompt_len} new={args.new_tokens}")
    results = {}
    for use_cache in (False, True):
        times = step_latencies(model, prompt, args.new_tokens, use_cache)
        results[use_cache] = times
        total = sum(times)
        print(f"  cache={'on ' if use_cache else 'off'}  total {total:7.2f}s  "
              f"{args.new_tokens * args.batch_size / total:8.1f} tok/s")

    print("\n  per-step latency (ms) at sequence length T")
    print(f"  {'T':>6} {'no cache':>10} {'cache':>10}")
    marks = [m for m in (1, 32, 64, 128, 256, 512, 1024, 2048) if m < args.new_tokens] + [args.new_tokens - 1]
    for m in marks:
        lo, hi = max(0, m - 2), min(args.new_tokens, m + 3)  # median of 5 neighbouring steps
        med = lambda xs: sorted(xs)[len(xs) // 2]  # noqa: E731
        print(f"  {args.prompt_len + m:>6} {1e3 * med(results[False][lo:hi]):>10.2f} {1e3 * med(results[True][lo:hi]):>10.2f}")
    speedup = sum(results[False]) / sum(results[True])
    print(f"\n  speed-up from the cache: {speedup:.1f}x")


if __name__ == "__main__":
    main()
