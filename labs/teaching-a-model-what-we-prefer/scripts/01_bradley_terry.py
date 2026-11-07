"""Section "From comparisons to a scalar field": where the sigmoid comes from, and what Bradley-Terry can't see.

1. Simulate annotators with Gumbel noise: the choice rate matches sigmoid(delta). Gaussian noise gives the probit.
2. Rescaled, the logistic and probit links differ by under 0.01.
3. Noise scale and reward scale are one knob: data from noise s = 2 gives fitted gaps half as large.
4. Shift invariance: adding a constant to every score leaves the loss unchanged to float32 rounding.
5. The gradient anneals itself: d/d(delta) of -log sigmoid(delta) is -sigmoid(-delta).
6. A preference cycle cannot be represented: the best fit predicts 0.5 for every pair.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from prefopt.bt import (best_logistic_rescale, bt_loss, fit_scores, probit, sample_comparisons,  # noqa: E402
                        simulate_choices)
from prefopt.report import Report  # noqa: E402


def main():
    rep = Report("01_bradley_terry")
    n = 200_000
    deltas = torch.tensor([-3.0, -1.5, -0.5, 0.0, 0.5, 1.2, 2.0, 3.0], dtype=torch.float64)

    rep.line(f"1. Annotator simulation, {n} annotators per reward gap")
    gum = simulate_choices(deltas, n, "gumbel", seed=0)
    gau = simulate_choices(deltas, n, "gaussian", seed=1)
    rows = [[d.item(), g.item(), torch.sigmoid(d).item(), a.item(), probit(d).item()]
            for d, g, a in zip(deltas, gum, gau)]
    rep.table(["delta", "Gumbel rate", "sigmoid", "Gaussian rate", "Phi(d/sqrt2)"], rows)
    err_g = (gum - torch.sigmoid(deltas)).abs().max().item()
    err_n = (gau - probit(deltas)).abs().max().item()
    se = 0.5 / n**0.5
    rep.line(f"largest |Gumbel rate - sigmoid| = {err_g:.4f}; |Gaussian rate - probit| = {err_n:.4f} "
             f"(one standard error at p = 0.5 is {se:.4f})")

    grid = torch.linspace(-6, 6, 1201)
    c, gap = best_logistic_rescale(grid)
    rep.line()
    rep.line(f"2. Best rescaled logistic sigmoid(c * d) vs probit: c = {c:.3f}, largest gap = {gap:.4f}")

    rep.line()
    rep.line("3. Noise scale vs reward scale: fit 20 item scores from 40k comparisons at two noise levels")
    r_true = torch.linspace(-2, 2, 20)
    fitted = {}
    for s in (1.0, 2.0):
        w, l = sample_comparisons(r_true, 40_000, scale=s, seed=2)
        f = fit_scores(w, l, 20)
        fitted[s] = f - f.mean()
        slope = torch.dot(fitted[s], r_true) / torch.dot(r_true, r_true)
        rep.line(f"   noise s = {s}: fitted scores = {slope.item():.3f} x true scores "
                 f"(corr {torch.corrcoef(torch.stack([fitted[s], r_true]))[0, 1].item():.4f})")
    ratio = (fitted[2.0].std() / fitted[1.0].std()).item()
    rep.line(f"   spread of the s = 2 fit / spread of the s = 1 fit = {ratio:.3f}")

    rep.line()
    w, l = sample_comparisons(r_true, 40_000, seed=3)
    f = fit_scores(w, l, 20)
    base = bt_loss(f[w], f[l]).item()
    shifted = bt_loss(f[w] + 7.25, f[l] + 7.25).item()
    rep.line(f"4. Shift invariance: loss {base:.10f}; after adding 7.25 to every score {shifted:.10f}")

    rep.line()
    rep.line("5. Self-annealing gradient: autograd d/d(delta) of -log sigmoid(delta), against -sigmoid(-delta)")
    d = torch.tensor([-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0], requires_grad=True)
    bt_loss(d, torch.zeros_like(d)).mul(d.numel()).backward()
    rep.table(["delta", "autograd", "-sigmoid(-delta)"],
              [[a.item(), g.item(), -torch.sigmoid(-a).item()] for a, g in zip(d.detach(), d.grad)])

    rep.line()
    rep.line("6. A preference cycle: A beats B, B beats C, C beats A, each 80% of the time (3000 labels per pair)")
    pairs = [(0, 1), (1, 2), (2, 0)]
    win, lose = [], []
    for a, b in pairs:
        k = 2400
        win += [a] * k + [b] * (3000 - k)
        lose += [b] * k + [a] * (3000 - k)
    win, lose = torch.tensor(win), torch.tensor(lose)
    s = fit_scores(win, lose, 3, init=torch.tensor([0.5, -0.3, 0.1]))
    preds = [torch.sigmoid(s[a] - s[b]).item() for a, b in pairs]
    rep.line(f"   fitted scores (centred) {[round(x, 4) for x in (s - s.mean()).tolist()]}")
    rep.line(f"   predicted P(A>B), P(B>C), P(C>A) = {[round(p, 4) for p in preds]}  (the data says 0.8 each)")

    rep.data = {"max_err_gumbel": err_g, "max_err_gaussian": err_n, "logistic_rescale": c, "logistic_probit_gap": gap,
                "spread_ratio_s2_over_s1": ratio, "loss": base, "loss_shifted": shifted, "cycle_predictions": preds}
    rep.save()


if __name__ == "__main__":
    main()
