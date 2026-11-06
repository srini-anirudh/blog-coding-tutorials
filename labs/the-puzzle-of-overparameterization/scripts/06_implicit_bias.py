"""Section 7: the algorithm picks the solution.

Part 1: gradient descent on unregularized logistic regression walks to the hard-margin SVM direction.
Part 2 (Lab 2): seven weight-decay settings, all at zero training error, give seven different answers.
Part 3: steepest descent in the l2, l1 and l-infinity geometries picks the max-margin solution in its own norm.

Run:
    python labs/the-puzzle-of-overparameterization/scripts/06_implicit_bias.py
"""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from overparam.common import Report, table
from overparam.implicit_bias import (angle, gd_logistic, hard_margin_svm, logistic_loss_grad, margin,
                                     separable_data, steepest_descent)

out = Report("06_implicit_bias")
cps = [10**k for k in range(1, 7)]
for seed, note in ((17, "next-closest point at 2.2x the support-vector margin"),
                   (1, "a near-support point at 1.08x the margin: slower")):
    X, y, _ = separable_data(40, seed=seed)
    w_svm = hard_margin_svm(X, y)
    path = gd_logistic(X, y, cps, lr=1.0)
    rows, prev = [], None
    for t in cps:
        w = path[t]
        loss, _ = logistic_loss_grad(w, X * y[:, None])
        rows.append({"step": t, "loss": loss, "norm": float(np.linalg.norm(w)),
                     "norm_gain_per_decade": None if prev is None else float(np.linalg.norm(w) - prev),
                     "angle_to_svm_rad": angle(w, w_svm), "angle_x_log_t": angle(w, w_svm) * math.log(t),
                     "l2_margin": margin(w, X, y), "train_err": float(np.mean(np.sign(X @ w) != y))})
        prev = float(np.linalg.norm(w))
    out(f"Part 1, dataset seed {seed} (n = 40, 2-D, {note})")
    out(f"SVM margin {margin(w_svm, X, y):.4f}; theory: norm grows by ||w_svm|| ln 10 = "
        f"{np.linalg.norm(w_svm) * math.log(10):.2f} per decade of steps, angle shrinks like 1 / log t")
    out(table(rows, {"step": "d", "loss": ".2e", "norm": ".2f", "norm_gain_per_decade": ".2f",
                     "angle_to_svm_rad": ".4f", "angle_x_log_t": ".3f", "l2_margin": ".4f", "train_err": ".2f"}))
    out()
    out.data[f"gd_seed_{seed}"] = {"svm_margin": margin(w_svm, X, y), "rows": rows}

X, y, w_true = separable_data(40, seed=17)
w_svm = hard_margin_svm(X, y)
Xt, yt, _ = separable_data(20000, seed=99, gap=0.0)
rows = []
for wd in (0, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1):
    w = gd_logistic(X, y, [1000], lr=1.0, weight_decay=wd)[1000]
    rows.append({"weight_decay": wd, "train_err": float(np.mean(np.sign(X @ w) != y)),
                 "angle_to_svm_rad": angle(w, w_svm), "angle_to_truth_rad": angle(w, w_true),
                 "test_acc": float(np.mean(np.sign(Xt @ w) == yt))})
out("Part 2 (Lab 2): 1,000 steps of gradient descent, same data, seven weight-decay settings")
out(table(rows, {"weight_decay": "g", "train_err": ".2f", "angle_to_svm_rad": ".3f", "angle_to_truth_rad": ".3f",
                 "test_acc": ".4f"}))
out()
out.data["weight_decay"] = rows

X, y, _ = separable_data(30, d=50, seed=2, gap=0.3, w_true=np.r_[np.ones(3), np.zeros(47)])
rows = []
for geom, name in (("l2", "normalized gradient descent (l2)"), ("l1", "greedy coordinate descent (l1)"),
                   ("linf", "sign descent (l-infinity)")):
    w = steepest_descent(X, y, geom, steps=20000, lr=0.02)
    a = np.sort(np.abs(w))[::-1]
    rows.append({"optimizer": name, "l2_margin": margin(w, X, y, 2), "l1_margin": margin(w, X, y, 1),
                 "linf_margin": margin(w, X, y, np.inf), "top5_share_of_l1": float(a[:5].sum() / a.sum()),
                 "train_err": float(np.mean(np.sign(X @ w) != y))})
w = hard_margin_svm(X, y)
a = np.sort(np.abs(w))[::-1]
rows.append({"optimizer": "exact l2 SVM", "l2_margin": margin(w, X, y, 2), "l1_margin": margin(w, X, y, 1),
             "linf_margin": margin(w, X, y, np.inf), "top5_share_of_l1": float(a[:5].sum() / a.sum()), "train_err": 0.0})
out("Part 3: n = 30, d = 50, 20,000 normalized steps of size 0.02. Margins are min_i y_i w.x_i / ||w||.")
out(table(rows, {"optimizer": "", "l2_margin": ".4f", "l1_margin": ".4f", "linf_margin": ".4f",
                 "top5_share_of_l1": ".3f", "train_err": ".2f"}))
out("Each optimizer wins on its own norm's margin. Coordinate descent is sparse; sign descent is dense.")
out.data["geometry"] = rows
out.save()
