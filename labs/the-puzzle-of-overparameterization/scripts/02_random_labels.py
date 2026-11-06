"""Section 2: networks memorize anything. Turn the label-corruption dial from 0 to 1 on a CIFAR-10 subset.

Same MLP (3072-512-512-10, 1.8M parameters, more than 300 per training image), same Adam, no weight decay,
dropout or augmentation. Only the labels change. Zhang et al. used CNNs on all 50,000 images; this is a
5,000-image subset with an MLP, so the accuracies are lower than theirs. The pattern is the point.

Run:
    python labs/the-puzzle-of-overparameterization/scripts/02_random_labels.py              # CIFAR-10, downloads once
    python labs/the-puzzle-of-overparameterization/scripts/02_random_labels.py --synthetic  # offline smoke run
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from overparam.common import Report, table
from overparam.memorize import corrupt_labels, fit, gaussian_pixels, load_cifar, synthetic_clusters

ap = argparse.ArgumentParser()
ap.add_argument("--synthetic", action="store_true")
ap.add_argument("--n-train", type=int, default=5000)
ap.add_argument("--n-test", type=int, default=2000)
ap.add_argument("--max-epochs", type=int, default=400)
args = ap.parse_args()

out = Report("02_random_labels" + ("_synthetic" if args.synthetic else ""))
if args.synthetic:
    x, y, xt, yt = synthetic_clusters(args.n_train, args.n_test)
    out(f"data: synthetic Gaussian clusters, {args.n_train} train / {args.n_test} test")
else:
    x, y, xt, yt = load_cifar(args.n_train, args.n_test)
    out(f"data: CIFAR-10, first {args.n_train} train / first {args.n_test} test images")

rows = []
for frac in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
    r = fit(x, corrupt_labels(y, frac), xt, yt, max_epochs=args.max_epochs)
    rows.append({"setting": f"labels corrupted {frac:.0%}", **r})
    out(table(rows[-1:], {"setting": "", "epochs_to_fit": "", "steps_to_fit": "", "train_acc": ".4f",
                          "test_acc": ".4f", "seconds": ""}).splitlines()[-1])
r = fit(gaussian_pixels(x), corrupt_labels(y, 1.0), gaussian_pixels(xt, seed=1), yt, max_epochs=args.max_epochs)
rows.append({"setting": "Gaussian pixels, random labels", **r})
out()
out(f"MLP with {rows[0]['params']:,} parameters, Adam lr 1e-3, batch 128, no explicit regularization")
out(table(rows, {"setting": "", "epochs_to_fit": "", "steps_to_fit": "", "train_acc": ".4f", "test_acc": ".4f",
                 "seconds": ""}))
real, rand = rows[0], rows[5]
if real["steps_to_fit"] and rand["steps_to_fit"]:
    out(f"random labels took {rand['steps_to_fit'] / real['steps_to_fit']:.1f}x the steps of real labels to fit")
out.data = {"rows": rows}
out.save()
