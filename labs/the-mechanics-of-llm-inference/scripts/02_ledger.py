"""Every number the post derives on paper, recomputed from the formulas in inferlab/ledger.py.

Each line shows the recomputed value next to the value printed in the post, and flags any mismatch beyond
rounding. Nothing here is measured: these are the post's own roofline bounds and byte counts, checked.

    python labs/the-mechanics-of-llm-inference/scripts/02_ledger.py --out .../results/ledger_check.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inferlab import ledger as lg  # noqa: E402
from inferlab.scheduling import POST_REQUESTS  # noqa: E402
from inferlab.timing import Report  # noqa: E402

GB, KIB = 1e9, 1024


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rep = Report(args.out)
    m8, m7, m70 = lg.MODELS["llama-3-8b"], lg.MODELS["llama-2-7b"], lg.MODELS["llama-3-70b"]
    h100 = lg.ACCELERATORS["h100"]
    mismatches = []

    def check(label: str, value: float, post: float, fmt: str = "{:,.3g}", rel: float = 0.02) -> None:
        ok = abs(value - post) <= rel * abs(post)
        if not ok:
            mismatches.append(label)
        rep(f"{'ok  ' if ok else 'DIFF'} {label:<62} {fmt.format(value):>12}   post: {fmt.format(post)}")

    rep("Section 3: prefill and decode")
    check("attention term = linear term at N_p (P = 8.03e9)", lg.attention_equals_linear(m8), 53_000)
    p_nonembed = m8.params - 2 * m8.embed_params  # untied: input table and LM head are both V x d
    check("same, with P excluding embedding table and LM head (6.98e9)", lg.attention_equals_linear(m8, p_nonembed),
          53_000)
    check("intensity, prefill 8192 x 4096 x 4096 (FLOP/byte)", lg.gemm_intensity(8192, 4096, 4096), 1638)
    check("intensity, decode 1 x 4096 x 4096 (FLOP/byte)", lg.gemm_intensity(1, 4096, 4096), 1.0)
    w = lg.prefill_flops(m8, 8192)
    check("prefill of 8,192 tokens, TFLOP", w / 1e12, 149)
    check("prefill time at 45% of peak, ms", w / (0.45 * h100.peak_flops) * 1e3, 335)
    check("decode step at 8k, GFLOP", lg.decode_flops(m8, 8192) / 1e9, 20.4)
    check("decode step at 8k, bytes GB", lg.decode_step_bytes(m8, 1, 8192) / GB, 17.2)
    t = lg.decode_step_floor(m8, h100, 1, 8192)
    check("decode step at 8k, floor ms", t * 1e3, 5.1)
    check("decode achieved % of peak", 100 * lg.decode_flops(m8, 8192) / t / h100.peak_flops, 0.40)

    rep("\nSection 4: the cache")
    without, with_cache = lg.no_cache_vs_cache_flops(m8.params, 1000, 500)
    check("no-cache / cache FLOPs, 1,000 prompt + 500 out", without / with_cache, 417)

    rep("\nSection 5: Table 2, KV per token (KiB) and at 8k / 32k / 128k (GB)")
    for key, kib, at in [("llama-2-7b", 512, (4.29, 17.2, 68.7)), ("llama-3-8b", 128, (1.07, 4.29, 17.2)),
                         ("mistral-7b", 128, (1.07, 4.29, 17.2)), ("llama-3-70b", 320, (2.68, 10.7, 42.9))]:
        m = lg.MODELS[key]
        check(f"{m.name} per token", lg.kv_bytes_per_token(m) / KIB, kib)
        for n, post in zip((8192, 32768, 131072), at):
            check(f"{m.name} at {n:,}", lg.kv_bytes(m, 1, n) / GB, post)

    rep("\nSections 7 and 8: MLA, crossover, logits")
    check("MLA elements per token per layer, DeepSeek-V2 (d_c 512 + 64)", lg.mla_bytes_per_token(1, 512, 64, 1), 576)
    check("GQA-8 elements per token per layer (2 x 8 x 128)", 2 * 8 * 128, 2048)
    check("crossover, Llama-3-8B, cached tokens", lg.crossover_tokens(m8), 122_500)
    check("crossover, 7B full MHA, cached tokens", lg.crossover_tokens(m7), 25_700)
    check("decode logits, batch 256, V 128k, fp32, MB", 256 * 128_256 * 4 / 1e6, 131)
    check("prefill logits, 8k x 128k, fp32, GB", 8192 * 128_256 * 4 / GB, 4.2)

    rep("\nSection 10: the decode wall")
    check("8B bf16 weight read on H100, ms", 2 * m8.params / h100.bandwidth * 1e3, 4.8)
    check("single-stream ceiling, tok/s", h100.bandwidth / (2 * m8.params), 209)
    check("KV at 128k vs weights (8B): KV GB", lg.kv_bytes(m8, 1, 131072) / GB, 17.2)

    rep("\nSection 11: batching, B* and Table 4")
    for key, post in [("h100", 295), ("a100", 153), ("l40s", 419)]:
        check(f"B* on {lg.ACCELERATORS[key].name}", lg.b_star(lg.ACCELERATORS[key]), post)
    for n, kv, bc, i_inf, pct, ceil in [(2048, 0.27, 60, 63.8, 21.6, 12_500), (8192, 1.07, 15, 19.0, 6.4, 3_120),
                                        (32768, 4.29, 3.7, 7.7, 2.6, 780), (131072, 17.2, 0.9, 4.9, 1.7, 195)]:
        check(f"N={n:,}: KV per sequence GB", lg.kv_bytes(m8, 1, n) / GB, kv)
        check(f"N={n:,}: B_cross", lg.b_cross(m8, n), bc, rel=0.05)
        check(f"N={n:,}: I at B -> inf", lg.decode_intensity_limit(m8, n), i_inf)
        check(f"N={n:,}: ceiling as % of peak", 100 * lg.decode_intensity_limit(m8, n) / lg.b_star(h100), pct)
        check(f"N={n:,}: throughput ceiling tok/s", h100.bandwidth / lg.kv_bytes(m8, 1, n), ceil)
    check("I at N -> inf, GQA-4, bf16 cache (2g/s)", 2 * m8.group / 2, 4)
    check("B_max at 8k, 80 GB, 2 GB workspace", lg.b_max(m8, h100, 8192), 57, rel=0.03)

    rep("\nSection 15: speculative decoding")
    check("E[tokens], alpha 0.8, gamma 4", lg.spec_expected_tokens(0.8, 4), 3.36)
    check("speed-up with draft cost 5%", lg.spec_speedup(0.8, 4, 0.05), 2.8)
    check("idle tensor-core share in decode, 1/295 as %", 100 / lg.b_star(h100), 0.34)

    rep("\nSection 16: distributed")
    single = 2 * m8.params / h100.bandwidth
    for n, post in [(1, 4.79), (2, 2.78), (4, 1.58), (8, 0.98)]:
        check(f"TP={n}, 8B batch 1, 6 us per all-reduce, ms", lg.tp_decode_step(single, n, m8.layers) * 1e3, post)
    for b, post in [(1, 2), (4, 5.5), (16, 7.9)]:
        check(f"distinct experts, E 8, k 2, batch {b}", lg.moe_distinct_experts(8, 2, b), post)

    rep("\nSection 17: VLM token and cache accounting (7B full-MHA backbone, P = 6.74e9)")
    vis = 13 * 729 / 4
    check("visual tokens, 13 tiles x 729 / 4", vis, 2370)
    check("KV of one image, GB", vis * lg.kv_bytes_per_token(m7) / GB, 1.24)
    check("KV of a 30-token question, MB", 30 * lg.kv_bytes_per_token(m7) / 1e6, 15.7)
    check("ViT FLOPs, So400m (0.4B, 27 layers, d 1152), TFLOP",
          (2 * 0.4e9 * 9477 + 4 * 27 * 729**2 * 1152 * 13) / 1e12, 8.4)
    check("LLM prefill over 2,400 tokens, TFLOP", lg.prefill_flops(m7, 2400) / 1e12, 36.6)

    rep("\nSection 19: static batching of the figure-15 requests")
    check("slot utilisation, %", 100 * lg.static_batch_utilisation([r[2] for r in POST_REQUESTS]), 29.5)

    rep(f"\n{len(mismatches)} lines differ from the post by more than rounding:")
    for label in mismatches:
        rep("  - " + label)
    _ = m70
    rep.save()


if __name__ == "__main__":
    main()
