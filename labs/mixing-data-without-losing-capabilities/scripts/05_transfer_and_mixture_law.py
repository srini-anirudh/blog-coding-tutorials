"""Objectives, a transfer matrix, and a fitted mixture law checked against stratified sampling (post sections 7-9).

Four sources (add, rev, sort, copy) and four capabilities (accuracy on the same four). Each run is 1,500 steps,
which leaves the harder skills unfinished, so the mixture decides the outcome. In order:

  0. the post's Model A / Model B table, and why a weighted sum cannot pick a point in a dent of the front
  1. stratified (uniform) baseline, 3 seeds: the noise floor
  2. leave-one-out ablations at two budgets (1,500 and 4,500 steps): one column per removed source
  3. 16 random mixtures + the runs above -> fit S_j(w) = sum_i c_ij w_i + b_j log(w_j + 0.02), clipped to [0, 100]
  4. check the fit on 4 held-out random mixtures
  5. optimise four objectives on the fitted surface, train each winner with 2 seeds, compare with stratified

    python labs/mixing-data-without-losing-capabilities/scripts/05_transfer_and_mixture_law.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mixlab import objectives as O  # noqa: E402
from mixlab.model import make_model  # noqa: E402
from mixlab.runs import Report, n_procs, parallel_map  # noqa: E402
from mixlab.train import evaluate, make_pools, scores, train  # noqa: E402

TASKS = ("add", "rev", "sort", "copy")
BUDGET = 1500
TR, EV = make_pools(TASKS)


def run(cfg):
    w, steps, seed = cfg
    model = make_model(seed=seed)
    train(model, TR, dict(zip(TASKS, w)), steps=steps, seed=seed)
    s = scores(evaluate(model, EV))
    return [s[t] for t in TASKS]


def fmt(v):
    return "(" + ", ".join(f"{x:.2f}" for x in v) + ")"


def avg(rows):
    return [round(sum(r[j] for r in rows) / len(rows), 1) for j in range(len(rows[0]))]


def main() -> None:
    rep = Report("05_transfer_and_mixture_law", workers=n_procs())

    rep("0a. the post's two models: same mean, different minimum")
    a, b = [95, 95, 94, 15], [78, 76, 75, 71]
    rep(f"    A {a}: mean {sum(a) / 4:.2f}, min {min(a)}    B {b}: mean {sum(b) / 4:.2f}, min {min(b)}")
    front = [(1.0, 0.0), (0.9, 0.3), (0.55, 0.55), (0.3, 0.9), (0.0, 1.0)]  # (0.55, 0.55) sits in a dent
    hits = O.linear_scalarisation_hits(front)
    rep(f"0b. non-convex front {front}: points a weighted sum ever picks: {sorted(front[i] for i in hits)}")
    rep(f"    max-min picks {max(front, key=min)}")
    rep("")

    uni = (0.25,) * 4
    loo = [tuple(0.0 if i == k else 1 / 3 for i in range(4)) for k in range(4)]
    probes = O.dirichlet_probes(4, 16, seed=0)
    held = O.dirichlet_probes(4, 4, seed=1)
    cfgs = [(uni, BUDGET, s) for s in (0, 1, 2)] + [(w, BUDGET, s) for w in loo for s in (0, 1)]
    cfgs += [(uni, 3 * BUDGET, s) for s in (0, 1)] + [(w, 3 * BUDGET, s) for w in loo for s in (0, 1)]
    cfgs += [(w, BUDGET, 0) for w in probes + held]
    res = dict(zip(cfgs, parallel_map(run, cfgs)))

    base = [res[(uni, BUDGET, s)] for s in (0, 1, 2)]
    mu = avg(base)
    sd = [round((sum((r[j] - mu[j]) ** 2 for r in base) / 2) ** 0.5, 1) for j in range(4)]
    rep("1. stratified baseline (uniform), 3 seeds, accuracy %")
    rep.table(["seed", *TASKS, "mean", "min"], [[s, *r, round(sum(r) / 4, 1), min(r)] for s, r in enumerate(base)])
    rep(f"   mean {mu}, seed std {sd}  <- differences smaller than about 2 std are ties")
    rep("")

    rep("2. leave-one-out transfer matrix: M[i][j] = S_j(full) - S_j(full without source i), points, mean of seeds 0, 1")
    for steps in (BUDGET, 3 * BUDGET):
        full = avg([res[(uni, steps, s)] for s in (0, 1)])
        rows, without = [], []
        for k, w in enumerate(loo):
            s = avg([res[(w, steps, sd)] for sd in (0, 1)])
            without.append(s)
            rows.append([f"remove {TASKS[k]}", *(f"{full[j] - s[j]:+.1f}" for j in range(4))])
        rep(f"   budget {steps} steps; full mixture scores {full}")
        rep.table(["source", *(f"effect on {t}" for t in TASKS)], rows)
        rep.data[f"loo_{steps}"] = {"full": full, "without": without}
    rep("")

    fit_ws = [uni, *loo, *probes]
    fit_ss = [mu, *rep.data[f"loo_{BUDGET}"]["without"], *(res[(w, BUDGET, 0)] for w in probes)]
    sur = O.Surrogate(4, lo=0.0, hi=100.0).fit(fit_ws, fit_ss)
    rep(f"3. surrogate fitted on {len(fit_ws)} runs. Linear part c_ij (row = source, column = capability):")
    rep.table(["source", *TASKS], [[TASKS[i], *(f"{c:+.1f}" for c in row)] for i, row in enumerate(sur.transfer_matrix())])
    rep("   own-data log coefficient b_j: " + ", ".join(f"{t} {float(sur.coef[j][-1]):+.1f}" for j, t in enumerate(TASKS)))
    in_err = [abs(p - s) for w, s in zip(fit_ws, fit_ss) for p, s in zip(sur.predict(w), s)]
    rep(f"   in-sample mean |error| {sum(in_err) / len(in_err):.1f} points")
    rep("")
    rep("4. held-out random mixtures: predicted vs actual")
    rows, errs = [], []
    for w in held:
        p, s = sur.predict(w), res[(w, BUDGET, 0)]
        errs += [abs(x - y) for x, y in zip(p, s)]
        rows.append([fmt(w), *(f"{x:.0f}/{y:.0f}" for x, y in zip(p, s))])
    rep.table(["mixture (add, rev, sort, copy)", *(f"{t} pred/act" for t in TASKS)], rows)
    rep(f"   held-out mean |error| {sum(errs) / len(errs):.1f} points")
    rep("")

    grid = O.simplex_grid(4, 0.05)
    floors = [m - 2 for m in mu]
    objs = {
        "mean": O.mean_obj,
        "max-min": O.maxmin_obj,
        "weighted u=(.4,.2,.2,.2)": O.weighted_obj([0.4, 0.2, 0.2, 0.2]),
        "max add s.t. others >= stratified-2": O.constrained_obj(0, floors),
    }
    picks = {name: O.best_on_grid(sur.predict, f, grid) for name, f in objs.items()}
    ver = [(w, BUDGET, s) for w, _ in picks.values() for s in (0, 1)]
    vres = parallel_map(run, ver)
    rep("5. objective -> mixture picked on the fitted surface -> real runs (2 seeds), accuracy %")
    rows = [["stratified (3 seeds)", fmt(uni), "-", *mu, round(sum(mu) / 4, 1), min(mu)]]
    for k, (name, (w, pred)) in enumerate(picks.items()):
        act = avg(vres[2 * k : 2 * k + 2])
        rows.append([name, fmt(w), "/".join(f"{x:.0f}" for x in pred), *act, round(sum(act) / 4, 1), min(act)])
        rep.data.setdefault("picks", {})[name] = {"w": w, "pred": pred, "actual": vres[2 * k : 2 * k + 2]}
    rep.table(["objective", "mixture (add, rev, sort, copy)", "predicted", *TASKS, "mean", "min"], rows)
    rep.data.update({"baseline": base, "probes": {str(w): res[(w, BUDGET, 0)] for w in probes + held}})
    rep.save()


if __name__ == "__main__":
    main()
