"""When more thinking hurts: the walk that produces an inverted U (post section 10, Figure 6).

A toy model, not a measurement of any LLM. A trace moves between 'working', 'correct' and
'captured by a distractor'. Skill fades with depth; the distractor's pull does not. With no pull,
accuracy only rises. With pull, it rises, peaks and falls, and the peak comes earlier for easy problems.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from ttc.report import Report
from ttc.walk import accuracy_curve, negative_flips, sample_walks

r = Report("08_inverted_u")
rng = np.random.default_rng(0)
T, TAU = 200, 30.0
marks = [5, 10, 20, 40, 80, 120, 200]
pulls = {"none": 0.0, "moderate": 0.004, "strong": 0.012}
skills = {"easy": 0.25, "hard": 0.05}

r(f"steps T = {T}, skill fades as exp(-t/{TAU:.0f}). Accuracy = P(correct state) at step t.")
r("problem pull       " + "  ".join(f"t={t:<4d}" for t in marks) + "  peak step  peak acc  final  neg flips")
data = {}
for sname, s in skills.items():
    for pname, d in pulls.items():
        acc = accuracy_curve(T, s, d, TAU)
        walks = sample_walks(20000, T, s, d, TAU, rng)
        mc = (walks[:, -1] == 1).mean()
        assert abs(mc - acc[-1]) < 0.02, "Monte Carlo disagrees with the exact propagation"
        peak = int(acc.argmax()) + 1
        flips = negative_flips(walks)
        r(f"{sname:<7} {pname:<10} " + "  ".join(f"{acc[t - 1]:6.3f}" for t in marks) +
          f"  {peak:9d}  {acc.max():8.3f}  {acc[-1]:5.3f}  {flips:9.3f}")
        data[f"{sname}/{pname}"] = {"acc": acc.round(4).tolist(), "peak_step": peak, "neg_flips": flips}
r("With no pull the curve is monotone. With pull the peak appears, and the easy problem peaks first.")
r("'neg flips' is the share of traces that reached the correct state and ended outside it.")
r.save({"T": T, "tau": TAU, "pulls": pulls, "skills": skills, "curves": data})
