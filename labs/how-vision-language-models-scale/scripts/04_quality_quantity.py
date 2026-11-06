"""The best data filter depends on the compute budget (Goyal et al., 2024): simulation, then a real run.

    python labs/how-vision-language-models-scale/scripts/04_quality_quantity.py
    python labs/how-vision-language-models-scale/scripts/04_quality_quantity.py --quick   # 1 seed, short

Part 1 evaluates the utility-decay model with illustrative constants (as the post's Figure 4 does).
Part 2 trains real students on a noisy pool filtered three ways and measures clean test accuracy.
"""

import argparse

import numpy as np
import torch
from _common import Report, save_json, table

from vlmscale.laws import decayed_error
from vlmscale.quality import filter_top, make_pool, train_student

ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true")
ap.add_argument("--seeds", type=int, default=3)
a = ap.parse_args()
torch.set_num_threads(2)
r = Report("quality_quantity_quick" if a.quick else "quality_quantity")

# ---------------------------------------------------------------- part 1: the model
r("== Part 1: utility-decay model, illustrative constants, 128M-sample pool, half-life tau = 3 repeats ==")
# pool size, a, b. Same intercept; cleaner pools have a steeper first-pass slope b (higher utility).
pools = {"top 10% (aggressive)": (12.8e6, 1.0, 0.20), "top 30% (mild)": (38.4e6, 1.0, 0.19),
         "no filter": (128e6, 1.0, 0.18)}
budgets = [1e7, 1e8, 3e8, 5e8, 1e9, 3e9, 1e10]
rows = []
for c in budgets:
    errs = {k: float(decayed_error(c, s, sc, b, tau=3.0)[0]) for k, (s, sc, b) in pools.items()}
    best = min(errs, key=errs.get)
    rows.append([f"{c:.0e}"] + [f"{e:.4f}" for e in errs.values()] + [best])
r(table(rows, ["samples seen"] + [f"error, {k}" for k in pools] + ["best"]))
r("The constants are chosen for illustration, not fitted. The point is the shape: the winner moves from the")
r("aggressive filter to no filter as the budget grows, because the small pool repeats until it stops teaching.")

# ---------------------------------------------------------------- part 2: a real run
pool_size = 30_000 if a.quick else 100_000
views = [10_000, 40_000, 160_000, 640_000] if a.quick else [10_000, 40_000, 160_000, 640_000, 2_560_000]
seeds = 1 if a.quick else a.seeds
r(f"\n== Part 2: real students, pool of {pool_size:,} labelled points, label noise 0.6 x (1 - quality) ==")
r("Teacher: fixed random MLP on R^24, 10 classes. Student: 3-layer MLP, width 256, Adam 1e-3, batch 256.")
results = {}
for frac, label in ((0.1, "top 10%"), (0.3, "top 30%"), (1.0, "no filter")):
    accs = []
    for seed in range(seeds):
        p = make_pool(pool_size, noise_max=0.6, seed=seed)
        x, y = filter_top(p, frac)
        kept_clean = p["clean"][p["q"].argsort(descending=True)[: len(x)]]
        noise = float((y != kept_clean).float().mean())
        hist = train_student(x, y, p["x_test"], p["y_test"], p["classes"], views, seed=seed)
        accs.append([h["test_acc"] for h in hist])
    results[label] = {"kept": len(x), "label_noise": noise, "acc": np.mean(accs, 0).tolist(),
                      "acc_per_seed": accs}
rows = []
for i, v in enumerate(views):
    accs = {k: res["acc"][i] for k, res in results.items()}
    best = max(accs, key=accs.get)
    rows.append([f"{v:,}"] + [f"{acc:.3f} ({v / results[k]['kept']:.0f} ep)" for k, acc in accs.items()] + [best])
r(table(rows, ["sample views"] + [f"{k}: {res['kept']:,} kept, noise {res['label_noise']:.2f}"
                                   for k, res in results.items()] + ["best"]))
r(f"Clean test accuracy, mean of {seeds} seed(s); 'ep' is epochs over the kept set. Per-seed values are in the JSON.")
save_json(r.path.stem, {"views": views, "pool_size": pool_size, "results": results})
r.save()
