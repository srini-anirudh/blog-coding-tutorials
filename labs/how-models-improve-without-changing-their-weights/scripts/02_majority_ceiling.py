"""The generator-verifier gap on a synthetic benchmark (post section 5, Figure 3).

Each problem has an answer distribution: the correct answer with chance p, one systematic wrong
answer with chance w, and the rest spread over many rare wrong answers. Coverage climbs towards
1 - pi0. Majority voting climbs towards P[modal answer correct] and stops. The gap between the two
limits is what a better selector could still recover.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from ttc.selection import selector_curves
from ttc.report import Report

r = Report("02_majority_ceiling")
rng = np.random.default_rng(0)
N, POOL = 300, 256


def make_problem():
    p = rng.beta(0.7, 2.0) * (rng.random() > 0.10)  # 10% hopeless
    w = rng.uniform(0, 0.6) * (1 - p)  # a systematic wrong answer, as strong as 0.6 of the rest
    probs = np.array([p, w, 1 - p - w])
    kind = rng.choice(3, size=POOL, p=probs)
    answers = [0 if c == 0 else (1 if c == 1 else int(rng.integers(2, 10**6))) for c in kind]
    q_mode_correct = p > w and p > 0  # rare answers each have negligible mass
    return {"truth": 0, "answers": answers, "scores": list(rng.random(POOL)), "p": p,
            "modal_correct_true": bool(q_mode_correct)}


problems = [make_problem() for _ in range(N)]
ks = [1, 2, 4, 8, 16, 32, 64, 128, 256]
curves = selector_curves(problems, ks, n_draws=20, rng=rng)
pi0 = np.mean([pr["p"] == 0 for pr in problems])
mode_lim = np.mean([pr["modal_correct_true"] for pr in problems])

r(f"{N} problems, {POOL} samples each. Selectors use random subsets of k samples (20 draws per problem).")
r("k     coverage  majority  random-pick")
for i, k in enumerate(ks):
    r(f"{k:<5d} {curves['coverage'][i]:8.3f}  {curves['majority'][i]:8.3f}  {curves['best_score'][i]:8.3f}")
r(f"\ncoverage limit 1 - pi0          = {1 - pi0:.3f}")
r(f"majority limit P[mode correct]  = {mode_lim:.3f}")
r(f"generator-verifier gap          = {1 - pi0 - mode_lim:.3f}")
r("Random-pick is best-of-k under a score that carries no signal: it stays at the mean p for every k.")
r.save({"ks": ks, "curves": curves, "coverage_limit": float(1 - pi0), "majority_limit": float(mode_lim)})
