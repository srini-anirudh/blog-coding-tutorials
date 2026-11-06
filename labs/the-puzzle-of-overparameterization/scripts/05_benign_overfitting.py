"""Section 6: benign overfitting. Where the min-norm interpolant puts the noise, and which spectra hide it.

Run:
    python labs/the-puzzle-of-overparameterization/scripts/05_benign_overfitting.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from overparam.benign import growing_tail, spectrum_sweep
from overparam.common import Report, table

out = Report("05_benign_overfitting")
rows = growing_tail([31, 40, 60, 100, 200, 460, 1000, 3000, 10000])
out("Figure 7 setting: n = 30, sigma = 0.4, six signal directions of variance 1, tail directions of variance 0.05.")
out("Min-norm interpolant split into its signal piece and its noise piece. Medians over 40 draws.")
out(table(rows, {"p": "d", "train_residual": ".1e", "risk_total": ".4f", "risk_signal_piece": ".4f",
                 "risk_noise_piece": ".4f", "noise_norm2_tail_share": ".3f", "largest_tail_coef": ".4f"}))
best = min(rows, key=lambda r: r["risk_total"])
out(f"noise level sigma^2 = {rows[0]['noise_level']:.2f}. Best total risk {best['risk_total']:.4f} at p = {best['p']}, "
    "while fitting every noisy label exactly.")
out("As p grows the noise piece keeps shrinking, but the tail also soaks up signal: the signal piece grows.")
out()

ns = [25, 50, 100, 200, 400]
sweep = spectrum_sweep(ns)
out("Noise part of the min-norm risk (sigma = 1) as n grows, mean over 5 draws. p = 20,000 unless stated.")
srows = [{"spectrum": name, **{f"n={r['n']}": r["noise_risk_mean"] for r in v}} for name, v in sweep.items()]
out(table(srows, {"spectrum": "", **{f"n={n}": ".4f" for n in ns}}))
out("Benign means a row falls towards 0 as n grows. Only the spike-plus-wide-flat-tail spectrum gets there at this scale.")
out.data = {"growing_tail": rows, "spectrum_sweep": sweep}
out.save()
