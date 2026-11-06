"""Visual tokens against LLM size, trained for real on a small task.

    python labs/how-vision-language-models-scale/scripts/06_tokens_vs_size.py              # full grid, about 30 min on a busy CPU
    python labs/how-vision-language-models-scale/scripts/06_tokens_vs_size.py --quick      # 2 runs, 300 steps

Each run trains a TwoTowerVLM from scratch on the glyph-board task. The 64 patch features are
average-pooled to V visual tokens (1, 4, 16 or 64) before the connector, which is the compression knob
Li et al. (2024) turned. The LLM width sets N_L. We then compare runs at roughly equal inference cost
N_L (Q + V), the post's Figure 8 trade-off, separately for the coarse and the reading question.
"""

import argparse
import itertools
import multiprocessing as mp

import torch
from _common import Report, save_json, table

from vlmscale import data
from vlmscale.model import TwoTowerConfig
from vlmscale.train import train


def run_one(args: tuple) -> dict:
    v, l_dim, seed, steps, threads = args
    torch.set_num_threads(threads)
    cfg = TwoTowerConfig(pool_tokens=v, l_dim=l_dim, vocab=data.VOCAB, max_text=data.TEXT_LEN)
    out = train(cfg, steps=steps, eval_every=max(steps // 6, 1), seed=seed)
    out.update({"V": v, "l_dim": l_dim, "seed": seed})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    grid = [(v, d) for v, d in itertools.product((1, 4, 16, 64), (32, 64, 128))]
    if a.quick:
        grid, a.steps, a.seeds = [(1, 64), (16, 64)], 300, 1
    jobs = [(v, d, s, a.steps, a.threads) for (v, d) in grid for s in range(a.seeds)]
    r = Report("tokens_vs_size_quick" if a.quick else "tokens_vs_size")
    r(f"{len(jobs)} runs, {a.steps} steps of batch 128 each, {a.jobs} in parallel with {a.threads} threads each.")
    with mp.get_context("spawn").Pool(a.jobs) as pool:
        runs = pool.map(run_one, jobs)

    q = data.TEXT_LEN - 1  # prompt tokens the LLM reads before answering
    rows, agg = [], {}
    for v, d in grid:
        rs = [x for x in runs if x["V"] == v and x["l_dim"] == d]
        n_l = rs[0]["params"]["llm"]
        col = sum(x["final"]["color_acc"] for x in rs) / len(rs)
        gly = sum(x["final"]["glyph_acc"] for x in rs) / len(rs)
        agg[(v, d)] = (n_l, col, gly)
        rows.append([v, d, f"{n_l / 1e3:.0f}K", f"{n_l * (q + v) / 1e6:.2f}", f"{rs[0]['train_flops_per_sample'] / 1e6:.1f}",
                     f"{col:.3f}", f"{gly:.3f}", " ".join(f"{x['final']['glyph_acc']:.2f}" for x in rs)])
    r(f"\nAccuracy after {a.steps} steps on 1,024 fresh test images per question (mean over {a.seeds} seed(s)).")
    r("Inference cost = LLM matmul params x (Q + V), Q = 2 prompt tokens. Chance is 0.125 for both questions.")
    r(table(rows, ["V", "LLM width", "N_L", "inference cost (M)", "train MFLOP/sample", "coarse acc",
                   "reading acc", "reading per seed"]))

    if not a.quick:
        r("\nAt roughly equal inference cost (the diagonal: wide LLM + few tokens ... narrow LLM + many tokens):")
        diag = [(1, 128), (16, 64), (64, 32)]
        r(table([[v, d, f"{agg[(v, d)][0] * (q + v) / 1e6:.2f}", f"{agg[(v, d)][1]:.3f}", f"{agg[(v, d)][2]:.3f}"]
                 for v, d in diag], ["V", "LLM width", "inference cost (M)", "coarse acc", "reading acc"]))
    save_json("tokens_vs_size_quick" if a.quick else "tokens_vs_size",
              {"steps": a.steps, "runs": [{k: x[k] for k in ("V", "l_dim", "seed", "params", "visual_tokens",
                                                              "train_flops_per_sample", "seconds", "history")}
                                          for x in runs]})
    r.save()


if __name__ == "__main__":
    main()
