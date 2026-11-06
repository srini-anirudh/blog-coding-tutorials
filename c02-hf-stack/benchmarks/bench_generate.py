"""Generation throughput on SmolLM2-135M: recompute everything vs KV cache vs model.generate.

Run: python c02-hf-stack/benchmarks/bench_generate.py [--device cpu] [--new 64]
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from hfstack.generation import generate_with_cache, greedy_no_cache
from hfstack.utils import BASE_LM, be_polite

be_polite()
parser = argparse.ArgumentParser()
parser.add_argument("--device", default="cpu")
parser.add_argument("--new", type=int, default=64)
args = parser.parse_args()

tok = AutoTokenizer.from_pretrained(BASE_LM)
model = AutoModelForCausalLM.from_pretrained(BASE_LM, dtype=torch.float32).to(args.device).eval()
ids = tok(["The backward pass computes gradients by"], return_tensors="pt")["input_ids"].to(args.device)


def timed(fn) -> tuple[torch.Tensor, float]:
    start = time.perf_counter()
    out = fn()
    if args.device == "mps":
        torch.mps.synchronize()
    return out, time.perf_counter() - start


timed(lambda: generate_with_cache(model, ids, max_new_tokens=4))  # warm up
print(f"SmolLM2-135M, prompt {ids.size(1)} tokens, {args.new} new tokens, device {args.device}, float32")
results = {}
for name, fn in {
    "no cache (recompute history)": lambda: greedy_no_cache(model, ids, args.new),
    "our loop with KV cache": lambda: generate_with_cache(model, ids, max_new_tokens=args.new),
    "model.generate": lambda: model.generate(ids, max_new_tokens=args.new, min_new_tokens=args.new, do_sample=False),
}.items():
    results[name], seconds = timed(fn)
    print(f"  {name:<30} {seconds:>7.2f} s {args.new / seconds:>8.1f} tokens/s")
tokens = list(results.values())
print("  all three produced the same tokens:", all(torch.equal(t, tokens[0]) for t in tokens))
