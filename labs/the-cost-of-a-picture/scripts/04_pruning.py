"""Which tokens survive, and which questions still work: six reduction policies on Qwen2-VL-2B.

Each 672 x 672 synthetic picture becomes 24 x 24 = 576 visual tokens (the LLaVA-1.5 count). It holds a large
coloured shape (scene question) and a small printed number in the top or bottom band (reading question).
Every policy reduces the visual tokens to the same budget before the language model; kept tokens keep their
original M-RoPE positions. The "attention" policy uses FastV's score (attention received from the last token at
layer 2), but its kept set is applied at the language model's input, not after layer 2 as in FastV.
"""

import argparse
import time
from collections import defaultdict

import torch
from _common import quiet_transformers, write

from pixelcost import qwen
from pixelcost.selection import POLICIES, keep_all
from pixelcost.synth import QUESTIONS, is_correct, make_sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n", type=int, default=48, help="pictures (half top-band, half bottom-band numbers)")
    ap.add_argument("--budgets", default="144,64")
    ap.add_argument("--policies", default=",".join(POLICIES))
    ap.add_argument("--layer", type=int, default=2)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    quiet_transformers()
    model, processor = qwen.load(device=args.device)
    budgets = [int(b) for b in args.budgets.split(",")]
    policies = args.policies.split(",")
    configs = [("full", 576)] + [(p, k) for k in budgets for p in policies]

    correct = defaultdict(int)                  # (policy, k, question, band) -> count
    row_attention = torch.zeros(24)
    kept_rows = defaultdict(list)               # (policy, k) -> mean row of kept tokens
    prefill_ms = defaultdict(list)
    t_start = time.time()
    for i in range(args.n):
        s = make_sample(1000 + i, band="top" if i % 2 == 0 else "bottom")
        enc_cache = {}
        for q in ("color", "number"):
            inputs = qwen.build_inputs(processor, s.image, QUESTIONS[q], args.device)
            if "feats" not in enc_cache:                           # one encoder pass per picture
                enc_cache["feats"] = qwen.encode(model, inputs)
            p = qwen.prepare(model, inputs, enc_cache["feats"])
            full_e, full_p = qwen.reduce(p, keep_all)
            scores = qwen.attention_received(model, full_e, full_p, args.layer)[p.image_slice].float().cpu()
            if q == "number":
                row_attention += scores.reshape(24, 24).sum(1) / scores.sum()
            for policy, k in configs:
                fn = keep_all if policy == "full" else POLICIES[policy]
                red = fn(p.embeds[0, p.image_slice], 24, 24, k, scores=scores, seed=i)
                e, pos = qwen.apply_reduction(p, red)
                qwen.sync(args.device); t0 = time.perf_counter()
                logits, cache = qwen.prefill(model, e, pos)
                qwen.sync(args.device); prefill_ms[k].append((time.perf_counter() - t0) * 1e3)
                toks = qwen.decode(model, logits, cache, int(pos.max()) + 1, 8, qwen.eos_ids(model))
                text = processor.tokenizer.decode(toks, skip_special_tokens=True)
                correct[(policy, k, q, s.band)] += is_correct(q, s, text)
                if q == "number":
                    kept_rows[(policy, k)].append((red.index // 24).float().mean().item())
        print(f"picture {i + 1}/{args.n} done, {time.time() - t_start:.0f} s", flush=True)

    n_band = {"top": (args.n + 1) // 2, "bottom": args.n // 2}
    lines = [f"Qwen2-VL-2B, {args.n} pictures of 672 x 672 (576 visual tokens), greedy answers of at most 8 tokens",
             f"{'policy':<10} {'kept':>5} {'colour':>7} {'number':>7} {'num top':>8} {'num bottom':>11} "
             f"{'mean kept row':>14} {'prefill ms':>11}"]
    table = []
    for policy, k in configs:
        c = sum(correct[(policy, k, "color", b)] for b in n_band) / args.n
        nt = correct[(policy, k, "number", "top")] / n_band["top"]
        nb = correct[(policy, k, "number", "bottom")] / max(1, n_band["bottom"])
        nn_ = sum(correct[(policy, k, "number", b)] for b in n_band) / args.n
        mr = sum(kept_rows[(policy, k)]) / len(kept_rows[(policy, k)])
        pm = sorted(prefill_ms[k])[len(prefill_ms[k]) // 2]
        table.append(dict(policy=policy, kept=k, color=c, number=nn_, number_top=nt, number_bottom=nb,
                          mean_kept_row=mr, prefill_ms_median=pm))
        lines.append(f"{policy:<10} {k:>5} {c:>7.2f} {nn_:>7.2f} {nt:>8.2f} {nb:>11.2f} {mr:>14.1f} {pm:>11.1f}")
    ra = (row_attention / args.n).tolist()
    lines.append(f"\nShare of layer-{args.layer} last-token attention on visual tokens, by image row (0 = top), "
                 "mean over pictures:")
    lines.append("  " + " ".join(f"{v:.3f}" for v in ra))
    lines.append(f"  top 12 rows {sum(ra[:12]):.3f}, bottom 12 rows {sum(ra[12:]):.3f}")
    lines.append("\nRow 11.5 is the centre of the grid. Prefill ms is the median per budget over all policies.")
    tag = args.tag or ("cuda" if args.device.startswith("cuda") else "cpu")
    write(f"04_pruning_{tag}", "\n".join(lines), {"rows": table, "row_attention": ra, "n": args.n},
          device=args.device)


if __name__ == "__main__":
    main()
