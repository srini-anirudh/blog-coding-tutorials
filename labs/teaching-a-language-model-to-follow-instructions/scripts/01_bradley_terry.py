"""Section "From comparisons to a scalar field": the sigmoid falls out of Gumbel noise, and a reward falls out of
comparisons, up to a shift and a scale.

    python labs/teaching-a-language-model-to-follow-instructions/scripts/01_bradley_terry.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prefopt.bt import (bt_grad_magnitude, bt_loss, bt_prob, best_logistic_match, choice_rate,  # noqa: E402
                        fit_bt, simulate_comparisons, thurstone_prob)
from prefopt.common import Report  # noqa: E402


def main() -> None:
    rep = Report("01_bradley_terry")
    gen = torch.Generator().manual_seed(0)
    n_mc = 400_000

    rep.print("1. Annotator noise -> link function. Monte Carlo choice rate vs the closed forms,", n_mc, "draws each.")
    rows = []
    for delta in [-2.0, -1.0, 0.0, 0.5, 1.0, 2.0, 4.0]:
        rows.append({"delta": delta,
                     "gumbel_mc": choice_rate(delta, "gumbel", n_mc, gen), "sigmoid": bt_prob(delta).item(),
                     "gaussian_mc": choice_rate(delta, "gaussian", n_mc, gen), "probit": thurstone_prob(delta).item()})
    rep.table(rows)
    worst = max(max(abs(r["gumbel_mc"] - r["sigmoid"]), abs(r["gaussian_mc"] - r["probit"])) for r in rows)
    rep.print(f"largest |Monte Carlo - closed form| = {worst:.4f} (binomial std at p=0.5 is {0.5 / n_mc**0.5:.4f})")
    k, gap = best_logistic_match()
    rep.print(f"best rescaled logistic sigmoid({k:.3f} * delta) vs probit: largest gap {gap:.4f} in probability")
    rep.data["link"] = {"rows": rows, "max_mc_error": worst, "logistic_slope": k, "logistic_probit_gap": gap}

    rep.print("\n2. Fit a reward from comparisons. 20 items, true rewards ~ N(0, 1), 20000 noisy comparisons.")
    r_true = torch.randn(20, generator=gen, dtype=torch.float64)
    r_true = r_true - r_true.mean()
    fits = []
    for scale in [0.5, 1.0, 2.0]:
        w, l = simulate_comparisons(r_true, 20_000, gen, scale=scale)
        s = fit_bt(20, w, l)
        slope = ((s * r_true).sum() / (r_true * r_true).sum()).item()
        corr = torch.corrcoef(torch.stack([s, r_true]))[0, 1].item()
        fits.append({"noise_scale": scale, "fitted_slope": slope, "1/scale": 1 / scale, "correlation": corr})
    rep.table(fits)
    rep.print("The ordering is recovered (correlation near 1). The scale is not: the fitted slope tracks 1/noise_scale.")
    rep.data["fits"] = fits

    w, l = simulate_comparisons(r_true, 20_000, gen)
    s = fit_bt(20, w, l)
    shifted = bt_loss(s + 3.7, w, l).item() - bt_loss(s, w, l).item()
    rep.print(f"\nshift invariance: loss(r + 3.7) - loss(r) = {shifted:.2e}")
    rep.data["shift_loss_change"] = shifted

    rep.print("\n3. The gradient anneals itself: |d/d delta of -log sigmoid(delta)| = sigmoid(-delta).")
    deltas = torch.tensor([-4.0, -2.0, 0.0, 2.0, 4.0, 8.0], dtype=torch.float64)
    g = bt_grad_magnitude(deltas)
    grows = [{"delta": d.item(), "autograd": gi.item(), "sigmoid(-delta)": torch.sigmoid(-d).item()}
             for d, gi in zip(deltas, g)]
    rep.table(grows)
    rep.data["grad"] = grows

    rep.print("\n4. A preference cycle. Annotators prefer A>B, B>C and C>A, each 90% of the time.")
    cyc_w, cyc_l = [], []
    for a, b in [(0, 1), (1, 2), (2, 0)]:
        cyc_w += [a] * 900 + [b] * 100
        cyc_l += [b] * 900 + [a] * 100
    cw, cl = torch.tensor(cyc_w), torch.tensor(cyc_l)
    sc = fit_bt(3, cw, cl)
    nll = bt_loss(sc, cw, cl).item()
    rep.print(f"fitted scores {[round(x, 4) for x in sc.tolist()]}, NLL {nll:.4f} (log 2 = 0.6931).")
    rep.print("The best single scalar is a tie: it predicts 50% on every pair, where the data says 90%.")
    rep.data["cycle"] = {"scores": sc.tolist(), "nll": nll}
    rep.save()


if __name__ == "__main__":
    main()
