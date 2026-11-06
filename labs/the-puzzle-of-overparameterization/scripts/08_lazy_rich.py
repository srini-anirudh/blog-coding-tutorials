"""Section 9: lazy and rich training. The output scale alpha, not the width alone, decides whether features move.

Run:
    python labs/the-puzzle-of-overparameterization/scripts/08_lazy_rich.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from overparam.common import Report, table
from overparam.lazy_rich import lazy_limit, train

out = Report("08_lazy_rich")
cols = {"alpha": ".4g", "width": "d", "train_mse": ".1e", "test_mse": ".4f", "rel_weight_change": ".4f",
        "kernel_change": ".1e"}
out("Teacher y = relu(x_1) - E[.], x in R^10, n = 200 train / 2,000 test, full-batch GD, lr 1, 4,000 steps.")
kernel = lazy_limit()
out(f"lazy limit (kernel interpolation with the initial tangent kernel, width 1024): test MSE {kernel:.4f}")
rows = [train(a) for a in (0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 100.0)]
out("Width 1024, varying the output scale alpha")
out(table(rows, cols))
out()
wrows = []
for m in (256, 1024, 4096):   # at m = 64 the two scalings coincide (8 / sqrt(64) = 1)
    wrows.append(train(1.0, m=m))
    wrows.append(train(8.0 / m**0.5, m=m))
wrows.insert(0, train(1.0, m=64))
out("Width sweep. alpha = 1 is the NTK parameterization; alpha = 8 / sqrt(m) is the mean-field scaling")
out(table(wrows, cols))
out("Under NTK scaling the weight change shrinks like 1 / sqrt(width); under mean-field scaling it does not.")
out.data = {"lazy_limit_test_mse": kernel, "alpha_sweep": rows, "width_sweep": wrows}
out.save()
