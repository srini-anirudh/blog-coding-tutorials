"""Sections "Pathology 2 · Goodhart's law, measured" and "Pathology 3 · Length", at toy scale.

A gold reward labels pairs drawn from the reference policy. A proxy reward model (a ReLU MLP) is fitted to those
labels. We then optimise against the proxy two ways, both computed exactly over a pool of 4000 completions:
best-of-n, and the KL-regularised optimum ref * exp(proxy / beta) / Z swept over beta (the target RL aims at).
We watch gold reward against d = sqrt(KL(pi || ref)), fit Gao et al.'s two functional forms, and track "length".

This is not the paper's experiment: no language model, no PPO, a 6-dimensional pool. It shows the mechanism only.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prefopt.goodhart import (fit_form, fit_proxy, make_world, optimisation_curves, pairwise_accuracy,  # noqa: E402
                              peak, sample_pairs)
from prefopt.report import Report  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="one proxy only (for tests)")
    args = ap.parse_args()
    rep = Report("04_overoptimization" + ("_quick" if args.quick else ""))
    world = make_world()
    rep.line(f"pool 4000 completions, 6 features; effective number of completions under ref "
             f"{1 / (world.ref**2).sum().item():.0f}")

    width, labels = 16, 1000
    win, los = sample_pairs(world, labels, seed=1)
    proxy = fit_proxy(world, win, los, width)
    acc = pairwise_accuracy(world, proxy)
    curves = optimisation_curves(world, proxy)
    rep.line(f"\nMain run: proxy width {width}, {labels} labels, held-out pairwise accuracy on ref pairs {acc:.3f}")
    for name, key, label in (("best-of-n", "bon", "n"), ("KL-regularised optimum", "kl", "beta")):
        rep.line(f"\n{name}: gold and proxy are measured as gains over ref; length is coordinate 0, gold peaks at +1")
        rows = [[q[label], q["d"], q["proxy"], q["gold"], q["length"]] for q in curves[key][:: 2 if key == "kl" else 2]]
        rep.table([label, "d=sqrt(KL)", "proxy gain", "gold gain", "length gain"], rows)
        a, b, r2 = fit_form(curves[key], "bon" if key == "bon" else "rl")
        p = peak(curves[key])
        form = "d (a - b d)" if key == "bon" else "d (a - b log d)"
        rep.line(f"peak gold {p['gold']:.3f} at d = {p['d']:.2f}; last point gold {curves[key][-1]['gold']:.3f} at "
                 f"d = {curves[key][-1]['d']:.2f}. Fit gold = {form}: a = {a:.3f}, b = {b:.3f}, R^2 = {r2:.3f}")
    rep.data["main"] = {"width": width, "labels": labels, "accuracy": acc, "curves": curves}

    if not args.quick:
        rep.line("\nSweep: proxy width x number of labels, 2 seeds each (seed = labels and init)")
        rows, sweep = [], []
        for labels in (1000, 3000, 10000):
            for width in (4, 16, 64):
                for seed in (1, 2):
                    win, los = sample_pairs(world, labels, seed=seed)
                    proxy = fit_proxy(world, win, los, width, seed=seed)
                    c = optimisation_curves(world, proxy)
                    pb, pk = peak(c["bon"]), peak(c["kl"])
                    res = {"labels": labels, "width": width, "seed": seed, "accuracy": pairwise_accuracy(world, proxy),
                           "bon_peak_d": pb["d"], "bon_peak_gold": pb["gold"], "kl_peak_d": pk["d"],
                           "kl_peak_gold": pk["gold"], "kl_last_gold": c["kl"][-1]["gold"],
                           "kl_peak_length": pk["length"], "kl_last_length": c["kl"][-1]["length"]}
                    sweep.append(res)
                    rows.append([labels, width, seed, res["accuracy"], pb["d"], pb["gold"], pk["d"], pk["gold"],
                                 res["kl_last_gold"], res["kl_last_length"]])
        rep.table(["labels", "width", "seed", "RM acc", "BoN peak d", "BoN peak gold", "KL peak d", "KL peak gold",
                   "gold at beta=0.32", "length at beta=0.32"], rows)
        rep.data["sweep"] = sweep
    rep.save()


if __name__ == "__main__":
    main()
