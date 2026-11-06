"""Fit the real sweep (sections 6, 7 and 10). Reads results/sweep.jsonl, writes fit_sweep.json and isoflop.png.

1. Approach 2: a parabola through each IsoFLOP valley, then a power law through the minima.
2. Approach 3: the parametric law L = E + A N^-alpha + B D^-beta, fitted to every run.
3. Section 7: the same runs refitted with Kaplan's non-embedding N. Only the x-axis changes.
4. Section 10: fit on the smaller budgets only, predict every run at the largest budget, compare.

    python labs/scaling-laws-from-first-principles/scripts/05_fit_sweep.py
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))

import numpy as np  # noqa: E402

from scalelab.allocation import optimal  # noqa: E402
from scalelab.fit import fit_parametric, isoflop_minimum  # noqa: E402

RAMP = ["#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]  # one blue hue, light to dark = small to large C


def load(path: Path) -> tuple[list[str], list[dict]]:
    lines = path.read_text().splitlines()
    return [x for x in lines if x.startswith("#")], [json.loads(x) for x in lines if x.startswith("{")]


def approach2(runs: list[dict], key: str = "n_total") -> tuple[dict, float, float]:
    """IsoFLOP minima per budget, and the exponents of N_opt and D_opt against C."""
    minima = {}
    for C in sorted({r["budget"] for r in runs}):
        rs = sorted((r for r in runs if r["budget"] == C), key=lambda r: r[key])
        n_opt, l_min = isoflop_minimum([r[key] for r in rs], [r["val_loss"] for r in rs])
        best = min(rs, key=lambda r: r["val_loss"])
        minima[C] = {"n_opt": n_opt, "d_opt": C / (6 * n_opt), "loss_min": l_min, "best_run": best["shape"],
                     "bracketed": rs[0][key] < n_opt < rs[-1][key]}
    cs = np.array(sorted(minima))
    a = float(np.polyfit(np.log(cs), np.log([minima[c]["n_opt"] for c in cs]), 1)[0]) if len(cs) > 1 else float("nan")
    return minima, a, 1 - a


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default=str(LAB / "results" / "sweep.jsonl"))
    args = ap.parse_args()
    header, runs = load(Path(args.sweep))
    budgets = sorted({r["budget"] for r in runs})
    print("\n".join(h.replace("# hardware:", "# sweep hardware:") for h in header[:1]))
    print("# command: python scripts/05_fit_sweep.py   (the fits run on CPU)\n")
    print(f"{len(runs)} runs at {len(budgets)} budgets; C = 6 N_total D; TinyStories, 4096-token BPE, seq 256\n")

    print(f"{'C':>7s} {'model':9s} {'N_total':>8s} {'N_ne':>8s} {'D':>7s} {'D/N':>6s} {'val loss':>9s} "
          f"{'tok/s':>6s} {'sec':>5s}")
    for r in sorted(runs, key=lambda r: (r["budget"], r["n_total"])):
        print(f"{r['budget']:7.0e} {r['shape']:9s} {r['n_total'] / 1e6:7.2f}M {r['n_nonembed'] / 1e6:7.2f}M "
              f"{r['tokens'] / 1e6:6.1f}M {r['tokens'] / r['n_total']:6.1f} {r['val_loss']:9.4f} "
              f"{r['tokens_per_second'] / 1e6:5.1f}M {r['train_seconds']:5.0f}")
    gpu_s = sum(r["train_seconds"] for r in runs)
    print(f"total training time {gpu_s / 60:.1f} min\n")

    out = {}
    print("1. Approach 2 (IsoFLOP), N = N_total")
    minima, a, b = approach2(runs)
    for C, m in minima.items():
        flag = "" if m["bracketed"] else "  (minimum outside the sweep: extrapolated)"
        print(f"   C = {C:.0e}: N_opt = {m['n_opt'] / 1e6:6.2f}M  D_opt = {m['d_opt'] / 1e6:6.1f}M  "
              f"D/N = {m['d_opt'] / m['n_opt']:6.1f}  L_min = {m['loss_min']:.4f}  best run {m['best_run']}{flag}")
    print(f"   N_opt ~ C^{a:.3f}, D_opt ~ C^{b:.3f}   (Chinchilla Approach 2 reported 0.49 / 0.51)\n")
    out["approach2_total"] = {"a": a, "b": b, "minima": {f"{c:.0e}": m for c, m in minima.items()}}

    N, D, L = (np.array([r[k] for r in runs], float) for k in ("n_total", "tokens", "val_loss"))
    law, obj = fit_parametric(N, D, L)
    pred = law(N, D)
    print("2. Approach 3 (parametric), N = N_total")
    print(f"   L = {law.E:.3f} + {law.A:.4g} N^-{law.alpha:.3f} + {law.B:.4g} D^-{law.beta:.3f}")
    print(f"   implied N* ~ C^{law.a:.3f}, D* ~ C^{law.b:.3f}; in-sample mean |error| "
          f"{np.mean(np.abs(pred - L)):.4f} nats, max {np.max(np.abs(pred - L)):.4f}\n")
    out["parametric_total"] = dataclasses.asdict(law)

    print("3. Section 7: identical runs, Kaplan's non-embedding N (and C = 6 N_ne D)")
    runs_k = [dict(r, budget_k=6 * r["n_nonembed"] * r["tokens"]) for r in runs]
    Nk = np.array([r["n_nonembed"] for r in runs_k], float)
    law_k, _ = fit_parametric(Nk, D, L)
    share = np.mean([1 - r["n_nonembed"] / r["n_total"] for r in runs])
    print(f"   embedding share of N_total across the ladder: {min(1 - Nk / N):.0%} to {max(1 - Nk / N):.0%} "
          f"(mean {share:.0%})")
    print(f"   L = {law_k.E:.3f} + {law_k.A:.4g} N_ne^-{law_k.alpha:.3f} + {law_k.B:.4g} D^-{law_k.beta:.3f}")
    print(f"   implied N_ne* ~ C_ne^{law_k.a:.3f}   versus   N_total* ~ C^{law.a:.3f} with total parameters")
    lo, hi = min(budgets), max(budgets)
    nk_lo, nk_hi = optimal(law_k, lo)[0], optimal(law_k, hi)[0]
    print(f"   Kaplan-style fit: N_ne* at C_ne = {lo:.0e} / {hi:.0e} is {nk_lo / 1e6:.2f}M / {nk_hi / 1e6:.2f}M\n")
    out["parametric_kaplan"] = dataclasses.asdict(law_k)

    if len(budgets) >= 3:
        held = budgets[-1]
        fit_runs = [r for r in runs if r["budget"] < held]
        test = [r for r in runs if r["budget"] == held]
        law_s, _ = fit_parametric(*(np.array([r[k] for r in fit_runs], float) for k in ("n_total", "tokens", "val_loss")))
        mins_s, a_s, _ = approach2(fit_runs)
        cs = np.array(sorted(mins_s))
        n_pred = float(np.exp(np.polyval(np.polyfit(np.log(cs), np.log([mins_s[c]["n_opt"] for c in cs]), 1),
                                         np.log(held))))
        print(f"4. Section 10: fit on C <= {budgets[-2]:.0e} only ({len(fit_runs)} runs), predict C = {held:.0e}")
        print(f"   {'model':9s} {'measured':>9s} {'predicted':>10s} {'error':>8s}")
        errs = []
        for r in sorted(test, key=lambda r: r["n_total"]):
            p = float(law_s(r["n_total"], r["tokens"]))
            errs.append(p - r["val_loss"])
            print(f"   {r['shape']:9s} {r['val_loss']:9.4f} {p:10.4f} {p - r['val_loss']:+8.4f}")
        print(f"   mean |error| {np.mean(np.abs(errs)):.4f} nats; extrapolation factor in compute "
              f"{held / budgets[-2]:.0f}x past the last fitted budget")
        print(f"   N_opt at {held:.0e}: Approach 2 extrapolated {n_pred / 1e6:.2f}M, parametric "
              f"{optimal(law_s, held)[0] / 1e6:.2f}M, measured {minima[held]['n_opt'] / 1e6:.2f}M")
        out["holdout"] = {"budget": held, "mean_abs_error": float(np.mean(np.abs(errs))),
                          "n_opt_pred_approach2": n_pred, "n_opt_pred_parametric": optimal(law_s, held)[0],
                          "n_opt_measured": minima[held]["n_opt"]}

    (LAB / "results" / "fit_sweep.json").write_text(json.dumps(out, indent=2))
    plot(runs, minima, a, LAB / "results" / "isoflop.png")


def plot(runs, minima, a, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=110)
    for i, C in enumerate(sorted(minima)):
        rs = sorted((r for r in runs if r["budget"] == C), key=lambda r: r["n_total"])
        x = np.log10([r["n_total"] for r in rs])
        y = [r["val_loss"] for r in rs]
        color = RAMP[i % len(RAMP)]
        xs = np.linspace(x.min(), x.max(), 100)
        ax1.plot(10 ** xs, np.polyval(np.polyfit(x, y, 2), xs), color=color, lw=2, label=f"C = {C:.0e}")
        ax1.plot(10 ** x, y, "o", color=color, ms=6, mec="white", mew=1.5)
        ax1.plot(minima[C]["n_opt"], minima[C]["loss_min"], "*", color=color, ms=12, mec="white")
    ax1.set_xscale("log")
    ax1.set_xlabel("parameters N_total")
    ax1.set_ylabel("validation loss (nats/token)")
    ax1.set_title("IsoFLOP valleys", loc="left")
    ax1.legend(frameon=False, fontsize=8)
    cs = np.array(sorted(minima))
    ns = np.array([minima[c]["n_opt"] for c in cs])
    ax2.plot(cs, ns, "o", color=RAMP[2], ms=8, mec="white", mew=1.5)
    fit = np.polyfit(np.log(cs), np.log(ns), 1)
    cc = np.logspace(np.log10(cs.min()), np.log10(cs.max()), 50)
    ax2.plot(cc, np.exp(np.polyval(fit, np.log(cc))), color=RAMP[2], lw=2)
    ax2.text(cs[0], ns[-1], f"N_opt ∝ C^{a:.2f}", fontsize=9, color="#333333")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("compute C = 6 N_total D (FLOPs)")
    ax2.set_ylabel("N_opt (valley minimum)")
    ax2.set_title("Compute-optimal frontier", loc="left")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.25, lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)


if __name__ == "__main__":
    main()
