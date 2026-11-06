"""Figure 10: truth density of an append-only store, closed form against simulation, and precision at fixed k.

    python labs/memory-is-more-than-context/scripts/03_staleness.py
"""

from __future__ import annotations

import statistics

from _common import Report

from memlab.staleness import precision_at_k, rho_append, rho_managed, simulate_store

A, LAM, STEPS, SEEDS = 6, 0.12, 100, 20  # Figure 10's defaults: 6 facts per step, lambda 0.12, 100 steps
out = Report("03_staleness")
data: dict = {"a": A, "lambda": LAM, "seeds": SEEDS, "density": [], "precision": []}

out(f"a={A} facts/step, lambda={LAM}/step, {SEEDS} simulated stores per recall setting")
out(f"{'t':>4}{'lambda t':>9}{'append closed':>15}{'append sim':>12}{'1/(lambda t)':>14}"
    f"{'recall .7 closed':>18}{'recall .7 sim':>15}{'records (append)':>18}")
sims = {rec: [simulate_store(STEPS, A, LAM, rec, seed=s) for s in range(SEEDS)] for rec in (0.0, 0.7)}
for t in (5, 10, 25, 50, 100):
    app = statistics.mean(run[t - 1]["rho"] for run in sims[0.0])
    man = statistics.mean(run[t - 1]["rho"] for run in sims[0.7])
    recs = statistics.mean(run[t - 1]["records"] for run in sims[0.0])
    row = {"t": t, "append_closed": rho_append(t, LAM), "append_sim": app, "inv_lt": 1 / (LAM * t),
           "managed_closed": rho_managed(t, LAM, 0.7), "managed_sim": man, "records": recs}
    data["density"].append({k: round(v, 4) for k, v in row.items()})
    out(f"{t:>4}{LAM * t:>9.1f}{row['append_closed']:>15.3f}{app:>12.3f}{row['inv_lt']:>14.3f}"
        f"{row['managed_closed']:>18.3f}{man:>15.3f}{recs:>18.0f}")
true_100 = statistics.mean(run[-1]["true"] for run in sims[0.0])
out(f"true facts at t=100: simulated mean {true_100:.1f}, closed form a/lambda (1 - e^-12) = {A / LAM:.1f}")

out("\nPrecision@k with nothing going stale: 6 relevant memories, k=6,")
out("relevant scores N(1.0, 0.4^2), irrelevant N(0.0, 0.4^2), 200 trials per size")
out(f"{'store size n':>13}{'precision@6':>13}")
for n in (200, 2000, 20000):
    p = precision_at_k(n, 6, 6, 1.0, 0.0, 0.4, trials=200)
    data["precision"].append({"n": n, "precision": round(p, 4)})
    out(f"{n:>13}{p:>13.3f}")
out.save(data)
