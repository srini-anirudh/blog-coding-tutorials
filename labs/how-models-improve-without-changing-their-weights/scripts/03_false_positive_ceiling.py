"""The false-positive ceiling (post section 6, Figure 4).

Resample until something passes the verifier, then pick a passer at random. If wrong samples pass
with rate phi, accuracy tends to p / (p + (1-p) phi) however many samples you draw.
We check the exact finite-k formula against Monte Carlo, and print the post's four ceilings.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from ttc.selection import fp_ceiling, verifier_accuracy, verifier_accuracy_mc
from ttc.report import Report

r = Report("03_false_positive_ceiling")
rng = np.random.default_rng(0)
ks = [1, 4, 16, 64, 256, 1024]

r("== The post's example, p = 0.05 ==")
for phi in (0.0, 0.01, 0.05, 0.20):
    r(f"phi = {phi:<5} ceiling = {fp_ceiling(0.05, phi):.3f}")

r("\n== Accuracy against k: exact formula / Monte Carlo (20,000 trials) ==")
r("p     phi    " + "  ".join(f"k={k:<11d}" for k in ks) + "  ceiling")
data = {}
for p in (0.02, 0.05, 0.20):
    for phi in (0.0, 0.01, 0.05, 0.20):
        ex = verifier_accuracy(p, phi, ks)
        mc = [verifier_accuracy_mc(p, phi, k, 20000, rng) for k in ks]
        r(f"{p:<5} {phi:<5}  " + "  ".join(f"{e:.3f}/{m:.3f}  " for e, m in zip(ex, mc)) +
          f"  {fp_ceiling(p, phi):.3f}")
        data[f"p={p},phi={phi}"] = {"exact": list(map(float, ex)), "mc": mc, "ceiling": fp_ceiling(p, phi)}
r("With phi = 0 accuracy equals coverage and keeps rising. With phi > 0 it flattens at the ceiling.")
r("The ceiling depends on phi / p, so the same verifier hurts more on harder problems.")
r.save({"ks": ks, "curves": data})
