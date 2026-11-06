"""Post section 4, "Deriving the KV cache": the cache is forced by causality, and it is exact.

1. Run a sequence through the model, then run it again with one more token appended. The keys and values of
   every earlier position are compared. Causality says they cannot change.
2. Greedy generation with the cache and without it gives the same tokens.
3. The work: rows pushed through the linear layers with and without the cache, and the wall time, for a
   SmolLM2-135M-shaped model with random weights on CPU. The post's 8B example (417x) is computed, not run.

    python labs/the-mechanics-of-llm-inference/scripts/01_kv_cache_exact.py --out .../results/kv_cache_exact_cpu.txt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inferlab.decoder import SHAPES, Decoder, greedy_generate  # noqa: E402
from inferlab.ledger import no_cache_vs_cache_flops  # noqa: E402
from inferlab.timing import Report  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="smollm2-135m")
    ap.add_argument("--prompt", type=int, default=100)
    ap.add_argument("--new", type=int, default=50)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rep = Report(args.out)
    torch.manual_seed(0)
    c = SHAPES[args.shape]
    model = Decoder.random(c, max_len=args.prompt + args.new + 1)
    model.allocate_cache(1)
    rep(f"{args.shape} shape, random fp32 weights, CPU")

    # 1. Causality: appending a token cannot change the keys and values before it.
    t = 32
    seq = torch.randint(0, c.vocab, (1, t + 1))
    model.forward(seq[:, :t], 0)
    k_short, v_short = model.k_cache[:, :, :, :t].clone(), model.v_cache[:, :, :, :t].clone()
    model.forward(seq, 0)
    k_long, v_long = model.k_cache[:, :, :, :t], model.v_cache[:, :, :, :t]
    diff = max((k_short - k_long).abs().max().item(), (v_short - v_long).abs().max().item())
    rep(f"\n1. K and V at positions 0..{t - 1}, after {t} tokens and after {t + 1} tokens, all {c.layers} layers:")
    rep(f"   max abs difference {diff:.3e}, bit-identical: {torch.equal(k_short, k_long) and torch.equal(v_short, v_long)}")
    other = seq.clone()
    other[0, t] = (other[0, t] + 1) % c.vocab
    model.forward(seq, 0)
    k_a = model.k_cache[:, :, :, : t + 1].clone()
    model.forward(other, 0)
    k_b = model.k_cache[:, :, :, : t + 1]
    same = torch.equal(k_a[:, :, :, :t], k_b[:, :, :, :t])
    rep(f"   two sequences of {t + 1} tokens that differ only in the last token: earlier keys bit-identical: {same};"
        f" last key differs: {not torch.equal(k_a[:, :, :, t], k_b[:, :, :, t])}")

    # 2. and 3. Generation with and without the cache.
    prompt = torch.randint(0, c.vocab, (1, args.prompt))
    times = {}
    out = {}
    for use_cache in (True, False):
        t0 = time.perf_counter()
        out[use_cache] = greedy_generate(model, prompt, args.new, use_cache=use_cache)
        times[use_cache] = time.perf_counter() - t0
    rep(f"\n2. greedy, {args.new} new tokens after a {args.prompt}-token prompt: cached and uncached tokens equal: "
        f"{torch.equal(out[True], out[False])}")
    rows_cache = args.prompt + (args.new - 1)
    rows_none = sum(args.prompt + i for i in range(args.new))
    p_lin = sum(p.numel() for n, p in model.named_parameters() if "embed" not in n or c.tied)
    rep(f"\n3. rows through the linear layers: no cache {rows_none:,}, cache {rows_cache:,} "
        f"({rows_none / rows_cache:.1f}x)")
    rep(f"   linear FLOPs at 2 x {p_lin / 1e6:.0f}M params per row: no cache {2 * p_lin * rows_none / 1e12:.2f} TFLOP, "
        f"cache {2 * p_lin * rows_cache / 1e12:.3f} TFLOP")
    rep(f"   wall time: no cache {times[False]:.2f} s, cache {times[True]:.2f} s ({times[False] / times[True]:.1f}x)")
    without, with_cache = no_cache_vs_cache_flops(8.03e9, 1000, 500)
    rep(f"\n   post's example, computed: 8B model, 1,000-token prompt, 500 new tokens: "
        f"{without:.2e} vs {with_cache:.2e} FLOP = {without / with_cache:.0f}x")
    rep.save()


if __name__ == "__main__":
    main()
