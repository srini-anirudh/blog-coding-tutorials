"""The noise floor of a benchmark: SE = sqrt(p (1 - p) / n), checked by simulation.

Reproduces the post's numbers for MathVista testmini (1,000), MM-Vet (218) and LLaVA-Bench (60)
at p = 0.6, then draws many benchmarks of each size from one fixed "true" model and shows the
spread of measured scores. Last, two models whose true accuracies differ by 2 points: how often
does a benchmark of each size rank them the wrong way round?

    python labs/measuring-what-a-vlm-can-do/scripts/03_noise_floor.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from benchlab.report import Report  # noqa: E402
from benchlab.stats import standard_error  # noqa: E402

r = Report("03_noise_floor")
p, trials = 0.6, 20000
g = torch.Generator().manual_seed(0)
r(f"accuracy p = {p}; {trials} simulated benchmarks per size; items independent and binary")
r(f"{'benchmark':28s} {'n':>5s} {'SE formula':>11s} {'SE simulated':>13s} {'95% half-width':>15s}")
for name, n in [("MathVista testmini", 1000), ("MM-Vet", 218), ("LLaVA-Bench (In-the-Wild)", 60),
                ("MMStar", 1500), ("POPE, one setting", 3000)]:
    scores = (torch.rand(trials, n, generator=g) < p).float().mean(1)
    se = standard_error(p, n)
    r(f"{name:28s} {n:5d} {100 * se:10.1f}pt {100 * scores.std().item():12.1f}pt {100 * 1.96 * se:14.1f}pt")

r("\nTwo models with true accuracy 0.60 and 0.62, scored on independent draws of n items:")
r(f"{'n':>5s} {'P(worse model scores >= better)':>32s}")
for n in [60, 218, 1000, 1500, 5000]:
    a = (torch.rand(trials, n, generator=g) < 0.62).float().mean(1)
    b = (torch.rand(trials, n, generator=g) < 0.60).float().mean(1)
    r(f"{n:5d} {(b >= a).float().mean().item():32.3f}")
r("\nScores of two models on the same items are correlated, which narrows the interval for their")
r("difference. 11_analyse.py measures that with a paired bootstrap on real MMStar replies.")
r.save()
