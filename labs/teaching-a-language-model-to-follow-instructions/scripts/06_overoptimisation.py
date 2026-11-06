"""Section "Pathology 2 · Goodhart's law, measured": gold reward rises, peaks and falls as a proxy is optimised.

A toy copy of Gao, Schulman and Hilton (2022). The paper's scale (reward models up to 3B parameters, a 6B gold
model) is not reproduced. The curve shapes and the fitted functional forms are.

    python labs/teaching-a-language-model-to-follow-instructions/scripts/06_overoptimisation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prefopt.common import Report  # noqa: E402
from prefopt.overopt import A, B, best_of_n_curve, fit_form, fit_proxy, tilt_curve  # noqa: E402

NS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
BETAS = [8, 4, 2, 1.5, 1, 0.7, 0.5, 0.35, 0.25, 0.18, 0.12, 0.08, 0.05, 0.03]
PROXIES = [  # (label pairs, hidden width; 0 = linear)
    (1000, 0), (1000, 4), (1000, 32), (300, 32), (3000, 32), (10000, 32),
]
MIN_ESS = 100  # drop tilt points where fewer than this many pool samples carry the weight


def peak(rows: list[dict]) -> tuple[float, float]:
    best = max(rows, key=lambda r: r["gold"])
    return best["d"], best["gold"]


def main() -> None:
    torch.set_num_threads(8)
    rep = Report("06_overoptimisation")
    g0 = -B.sum().item()  # E[gold] under the reference: E[x] = 0, E[x_k^2] = 1
    best_gold = (A**2 / (4 * B)).sum().item() - g0
    rep.print("x ~ N(0, I_4) under the reference. Gold r*(x) = a.x - b.x^2.")
    rep.print("Gold is reported relative to its exact mean under the reference, so every curve starts at 0.")
    rep.print(f"The best single completion scores {best_gold:.2f} on that scale. d = sqrt(KL).")
    rows, curves = [], {}
    for n_pairs, width in PROXIES:
        proxy = fit_proxy(n_pairs, width)
        bo = best_of_n_curve(proxy, NS, trials=500)
        tl = [r for r in tilt_curve(proxy, BETAS) if r["ess"] >= MIN_ESS]
        for r in bo + tl:
            r["gold"] -= g0
            r["proxy_gain"] = r["proxy"] - bo[0]["proxy"]
        d_bo, g_bo = peak(bo)
        d_tl, g_tl = peak(tl)
        pos = [r for r in bo if r["d"] > 0]
        a_bo, b_bo, r2_bo = fit_form(torch.tensor([r["d"] for r in pos]), torch.tensor([r["gold"] for r in pos]), "bo")
        a_rl, b_rl, r2_rl = fit_form(torch.tensor([r["d"] for r in tl]), torch.tensor([r["gold"] for r in tl]), "rl")
        name = f"{n_pairs} pairs, " + ("linear" if width == 0 else f"width {width}")
        curves[name] = {"best_of_n": bo, "tilt": tl}
        rows.append({"proxy": name, "BoN peak d": d_bo, "BoN peak gold": g_bo, "BoN gold at n=16384": bo[-1]["gold"],
                     "tilt peak d": d_tl, "tilt peak gold": g_tl, "fit R^2 bo": r2_bo, "fit R^2 rl": r2_rl,
                     "alpha_bo": a_bo, "beta_bo": b_bo, "alpha_rl": a_rl, "beta_rl": b_rl})

    rep.print("\nOne best-of-n curve in full (1000 pairs, linear proxy): proxy keeps rising, gold turns over.")
    rep.table([{k: r[k] for k in ("n", "d", "proxy_gain", "gold")} for r in curves["1000 pairs, linear"]["best_of_n"]],
              "{:.3f}")
    rep.print("\nPeaks and fits for every proxy. 'tilt' is the exact KL-regularised optimum against the proxy,")
    rep.print(f"by reweighting 400k reference samples, kept while the effective sample size is >= {MIN_ESS}.")
    rep.table([{k: r[k] for k in ("proxy", "BoN peak d", "BoN peak gold", "BoN gold at n=16384", "tilt peak d",
                                  "tilt peak gold")} for r in rows], "{:.2f}")
    rep.print("\nGao et al.'s forms fitted by least squares: R_bo = d (alpha - beta d), R_rl = d (alpha - beta log d).")
    rep.table([{k: r[k] for k in ("proxy", "alpha_bo", "beta_bo", "fit R^2 bo", "alpha_rl", "beta_rl", "fit R^2 rl")}
               for r in rows], "{:.3f}")
    rep.print("\nThe tilt is the exact target of KL-regularised RL, not an RL run. The paper's finding that RL")
    rep.print("optimises and over-optimises more slowly than best-of-n is about RL's optimiser, so it is not tested.")
    rep.data.update({"summary": rows, "curves": curves, "best_possible_gold_vs_ref_mean": best_gold})
    rep.save()


if __name__ == "__main__":
    main()
