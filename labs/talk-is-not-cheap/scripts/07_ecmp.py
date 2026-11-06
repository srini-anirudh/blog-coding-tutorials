"""Section 15: ECMP hash collisions as balls in bins, and what packet spraying does to them.

Monte Carlo, 20,000 trials per row. No network is involved.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from commcost import fabric  # noqa: E402
from commcost.report import Report, table  # noqa: E402

R = Report("07_ecmp", note="Monte Carlo, numpy, seed 0")
rng = np.random.default_rng(0)
TRIALS = 20_000

R("k flows hashed onto k uplinks. A collective runs at the pace of the busiest uplink.")
rows, data = [], []
for k in (8, 16, 64, 256, 1024):
    m = fabric.ecmp_max_load(k, k, TRIALS, rng)
    spray = fabric.spray_max_load(k, k, 1000, 2000, rng)
    rows.append([k, f"{m.mean():.2f}", int(np.median(m)), f"{np.mean(m >= 2):.0%}", f"{fabric.ln_over_lnln(k):.2f}",
                 f"{spray.mean():.3f}"])
    data.append({"k": k, "ecmp_mean_max": float(m.mean()), "p_collision": float(np.mean(m >= 2)),
                 "ln_over_lnln": fabric.ln_over_lnln(k), "spray_mean_max": float(spray.mean())})
for line in table(rows, ["k", "mean busiest", "median", "P(any collision)", "ln k / ln ln k", "sprayed (1000 pkts)"]):
    R(line)
R()
R("The post's ln k / ln ln k is the leading term of an asymptotic result. At k = 64 it gives 2.92; the simulated")
R("busiest uplink carries about 4 flows on average. So the real penalty at this size is worse than the post quotes.")
R("Collisions are near certain at every k shown: the busiest link almost never carries just one flow.")
R("Spraying 1000 packets per flow brings the busiest link to within a few percent of an even share.")
R("Rail-aligned traffic needs no hash at all, so its busiest link carries exactly one flow by construction.")
R.save(data)
