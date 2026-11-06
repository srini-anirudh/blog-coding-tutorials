"""One FFN became hundreds: active vs total parameters, routing combinatorics, and load balancing.

Post sections: "One FFN became hundreds", "Why many small experts beat few large ones", "The problem MoE creates".
Part 3 is a toy: a 16-expert, top-2 MoE layer fit to a clustered regression task, small enough for a CPU. It shows
the rich-get-richer feedback and the two fixes. It does not reproduce DeepSeek-V3's results.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from organs.arith import PRESETS, calc_model, routing_configurations
from organs.moe import train_moe
from organs.report import Report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=2)
    args = ap.parse_args()
    rep = Report("08_moe")

    rep("1. Same total and active expert budget, different granularity:")
    a, b = routing_configurations(8, 2), routing_configurations(256, 8)
    rep(f"   C(8, 2) = {a}     C(256, 8) = {b:,} = {b:.2e}     ratio {b / a:.1e}")
    rep("")

    rep("2. Total and active parameters from the post's Figure 10 formulas (norms and biases not counted).")
    rep("   'Post' is the post's MoE table, which quotes the model reports.")
    reported = {"qwen3": ("235B", "22B"), "gptoss": ("117B", "5.1B"), "dsv3": ("671B", "37B")}
    rows = []
    for key in ("qwen3", "gptoss", "dsv3"):
        m = calc_model(PRESETS[key])
        rows.append([PRESETS[key].label, f"{m['total'] / 1e9:.1f}B", f"{m['active'] / 1e9:.1f}B", f"{m['frac']:.1%}",
                     f"{reported[key][0]} / {reported[key][1]}", f"{6 * m['active']:.2e}"])
    rep.table(["model", "total", "active", "fraction", "post", "train FLOPs/token = 6 N_active"], rows)
    rep("   The formulas land within about 2% of the reported numbers; they skip norms, biases and DeepSeek-V3's")
    rep("   multi-token-prediction module.")
    rep("")

    rep("3. Load balancing on a toy task: 16 experts, top-2, d = 32, 600 Adam steps. 'head start' adds a")
    rep("   learnable +4 to expert 0's router score at init, the 'slightly better early' case from the post.")
    rep("   load = share of routing slots on the test set. max/mean 1.0 is perfect, 8.0 means one expert is in")
    rep("   every token's top-2. dead = experts below 10% of the mean load.")
    rows = []
    for hs in (0.0, 4.0):
        for balance in ("none", "aux", "bias"):
            res = [train_moe(balance, s, head_start=hs) for s in range(args.seeds)]
            mean = lambda key: sum(r[key] for r in res) / len(res)
            rows.append(["yes" if hs else "no", {"none": "none", "aux": "aux loss 0.01",
                                                 "bias": "aux-free bias"}[balance],
                         f"{mean('max_over_mean'):.2f}", f"{mean('min_over_mean'):.2f}", f"{mean('dead'):.1f}",
                         f"{mean('entropy_frac'):.3f}", f"{mean('test_mse'):.4f}"])
            print(f"done {hs} {balance}", flush=True)
    rep.table(["head start", "balancing", "max/mean", "min/mean", "dead", "load entropy / log E", "test MSE"], rows)
    rep(f"   Means over {args.seeds} seeds. The bias rule: b_e += 0.01 * sign(1/E - load_e) after every step,")
    rep("   used only to choose the top-k, never in the gate weights or the loss.")
    print(f"saved {rep.save()}")


if __name__ == "__main__":
    main()
