"""Section 10: grokking on (a + b) mod 97 with 30% of the pairs for training.

Run:
    python labs/the-puzzle-of-overparameterization/scripts/09_grokking.py
    python labs/the-puzzle-of-overparameterization/scripts/09_grokking.py --steps 3000   # quick look
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from overparam.common import Report, table
from overparam.grokking import train

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=20000)
args = ap.parse_args()

out = Report("09_grokking")
out("MLP on shared 128-d embeddings, hidden 512, full-batch AdamW lr 1e-3, betas (0.9, 0.98)")
cols = {"step": "d", "train_acc": ".3f", "val_acc": ".3f", "weight_norm": ".1f", "fourier_top8": ".3f"}
for wd in (1.0, 0.0):
    rows, first = train(weight_decay=wd, steps=args.steps)
    out(f"weight decay {wd}: train acc >= 99% at step {first['train']}, val acc >= 99% at step {first['val']}")
    out(table(rows, cols))
    out()
    out.data[f"wd_{wd}"] = {"first_99": first, "rows": rows}
out("A random 97 x 128 table puts about 8/48 = 0.17 of its power in its top 8 frequencies.")
out.save()
