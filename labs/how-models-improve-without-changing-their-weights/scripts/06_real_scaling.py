"""Coverage, selection and verifier ceilings on real samples (post sections 4 to 6, Figures 2 to 4).

Reads results/05_samples.json (written by script 05) and measures, for k = 1 .. 64:
coverage (unbiased pass@k), majority vote, best-of-k by the model's own mean log-prob, and
'pick a random passer' under two imperfect verifiers. It fits a Beta prior to the per-problem
success counts and compares the power law it predicts with the measured failure curve.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from ttc.arith import last_digit_check, shape_check
from ttc.coverage import beta_uncovered, fit_beta_binomial, loglog_slope, pass_at_k
from ttc.report import RESULTS, Report
from ttc.selection import modal_answer_correct, selector_curves, verifier_accuracy

ap = argparse.ArgumentParser()
ap.add_argument("--samples", default=str(RESULTS / "05_samples.json"))
ap.add_argument("--draws", type=int, default=50)
args = ap.parse_args()
d = json.loads(Path(args.samples).read_text())
probs, K = d["problems"], d["k"]
r = Report("06_real_scaling")
r(f"samples: {d['model']}, {len(probs)} problems x {K} samples, temperature {d['temperature']}")
r(f"samples were generated on: {d['hardware'].removeprefix('hardware: ')}")
rng = np.random.default_rng(0)
ks = [k for k in (1, 2, 4, 8, 16, 32, 64) if k <= K]

c = np.array([sum(a == p["truth"] for a in p["answers"]) for p in probs])
p_hat = c / K
r(f"\nmean per-sample accuracy p = {p_hat.mean():.3f}; problems never solved in {K} samples: "
  f"{(c == 0).sum()}/{len(probs)}")

r("\n== Coverage law ==")
a, b = fit_beta_binomial(c, K)
fail = [1 - np.mean([pass_at_k(K, int(ci), k) for ci in c]) for k in ks]
pred = [beta_uncovered(a, b, k) for k in ks]
r(f"Beta-binomial fit to the success counts: a = {a:.3f}, b = {b:.3f}")
r("k      1 - C(k) measured   Beta(a,b) prediction")
for k, f, q in zip(ks, fail, pred):
    r(f"{k:<6d} {f:18.3f} {q:20.3f}")
tail = [k for k in ks if k >= 4]
s_meas = loglog_slope(tail, [f for k, f in zip(ks, fail) if k >= 4])
r(f"log-log slope for k >= 4: measured {s_meas:+.3f}, fitted -a = {-a:+.3f}")

r("\n== Selectors (random size-k subsets of the pool) ==")
for p in probs:
    p["passes"] = [last_digit_check(x, p["truth"]) for x in p["answers"]]
cur_ld = selector_curves(probs, ks, args.draws, rng)
for p in probs:
    p["passes"] = [shape_check(x, p["truth"]) for x in p["answers"]]
cur_sh = selector_curves(probs, ks, args.draws, rng)


def fpr(check):
    wrong = [(x, p["truth"]) for p in probs for x in p["answers"] if x != p["truth"]]
    return np.mean([check(x, t) for x, t in wrong])


phi_ld, phi_sh = fpr(last_digit_check), fpr(shape_check)


def predicted(check):
    """Mean over problems of the exact finite-k formula with that problem's own p and phi."""
    out = []
    for k in ks:
        vals = []
        for p in probs:
            wrong = [x for x in p["answers"] if x != p["truth"]]
            phi = np.mean([check(x, p["truth"]) for x in wrong]) if wrong else 0.0
            vals.append(verifier_accuracy(sum(x == p["truth"] for x in p["answers"]) / K, phi, k))
        out.append(float(np.mean(vals)))
    return out


pred_ld, pred_sh = predicted(last_digit_check), predicted(shape_check)
r(f"verifier false-positive rate on wrong samples: last-digit {phi_ld:.3f}, last-digit+length {phi_sh:.3f}")
r("k      coverage  majority  best-logprob  last-digit (formula)  digit+length (formula)")
for i, k in enumerate(ks):
    r(f"{k:<6d} {cur_ld['coverage'][i]:8.3f}  {cur_ld['majority'][i]:8.3f}  {cur_ld['best_score'][i]:12.3f}  "
      f"{cur_ld['verifier'][i]:10.3f} ({pred_ld[i]:.3f})  {cur_sh['verifier'][i]:12.3f} ({pred_sh[i]:.3f})")
mode_lim = np.mean([modal_answer_correct(p["answers"], p["truth"]) for p in probs])
cov_lim = np.mean(c > 0)
r(f"\nP[modal answer of all {K} samples is correct] = {mode_lim:.3f}  (the majority-vote limit)")
r(f"fraction with at least one correct sample     = {cov_lim:.3f}  (a lower bound on 1 - pi0)")
r(f"generator-verifier gap at k = {K}               = {cur_ld['coverage'][-1] - cur_ld['majority'][-1]:.3f}")

r("\n== Stratified by difficulty level ==")
r("level             p      coverage@64  majority-limit")
levels = sorted({p["level"] for p in probs})
strat = []
for lv in levels:
    sub = [i for i, p in enumerate(probs) if p["level"] == lv]
    row = (probs[sub[0]]["level_name"], p_hat[sub].mean(), np.mean(c[sub] > 0),
           np.mean([modal_answer_correct(probs[i]["answers"], probs[i]["truth"]) for i in sub]))
    strat.append(row)
    r(f"{row[0]:<18}{row[1]:.3f}  {row[2]:11.3f}  {row[3]:14.3f}")
for name, sel in (("levels 0-3", [i for i, p in enumerate(probs) if p["level"] < 4]),
                  ("levels 4-7", [i for i, p in enumerate(probs) if p["level"] >= 4])):
    f = [1 - np.mean([pass_at_k(K, int(c[i]), k) for i in sel]) for k in tail]
    ok = [(k, v) for k, v in zip(tail, f) if v > 0]
    s = loglog_slope(*zip(*ok)) if len(ok) >= 2 else float("nan")
    r(f"{name}: failure at k=1 {1 - p_hat[sel].mean():.3f}, at k={K} {f[-1]:.3f}, log-log slope {s:+.3f}")
r("The slope depends on which problems are in the benchmark, as Nuance 07 says.")
r.save({"ks": ks, "beta_fit": [a, b], "failure": fail, "failure_pred": pred, "selectors_last_digit": cur_ld,
        "selectors_shape": cur_sh, "verifier_pred_last_digit": pred_ld, "verifier_pred_shape": pred_sh,
        "phi": [float(phi_ld), float(phi_sh)], "majority_limit": float(mode_lim), "coverage_lb": float(cov_lim)})
