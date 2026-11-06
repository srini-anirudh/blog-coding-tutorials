"""Figure 13: the experience-following fixed point, closed form against a simulated agent loop.

    python labs/memory-is-more-than-context/scripts/04_poisoning.py
"""

from __future__ import annotations

import statistics

from _common import Report

from memlab.poisoning import e_star, iterate, simulate

EPS0, SEEDS, TASKS = 0.15, 20, 40000
out = Report("04_poisoning")
data: dict = {"eps0": EPS0, "seeds": SEEDS, "tasks": TASKS, "rows": []}

out("Post's worked numbers (eps0 = 0.15, phi = 0.9):")
for rho, claim in ((0.25, "roughly 0.02"), (1.0, "eps0 = 0.15"), (4.0, "roughly 0.89")):
    out(f"  rho={rho:<5} e* = {e_star(rho, 0.9, EPS0):.4f}   post says {claim}")

out(f"\nSimulated loop: {TASKS} tasks per run, error over the last half, {SEEDS} seeds per row")
out(f"{'phi':>5}{'rho':>6}{'e* closed':>11}{'iterated':>10}{'sim mean':>10}{'sim min':>9}{'sim max':>9}")
for phi in (0.8, 0.9):
    for rho in (0.25, 0.5, 1.0, 2.0, 4.0):
        runs = [simulate(rho, phi, EPS0, tasks=TASKS, seed=s)["error"] for s in range(SEEDS)]
        row = {"phi": phi, "rho": rho, "closed": e_star(rho, phi, EPS0), "iterated": iterate(rho, phi, EPS0),
               "sim_mean": statistics.mean(runs), "sim_min": min(runs), "sim_max": max(runs)}
        data["rows"].append({k: round(v, 4) for k, v in row.items()})
        out(f"{phi:>5}{rho:>6}{row['closed']:>11.4f}{row['iterated']:>10.4f}{row['sim_mean']:>10.4f}"
            f"{row['sim_min']:>9.4f}{row['sim_max']:>9.4f}")
worst = max(data["rows"], key=lambda r: abs(r["sim_mean"] - r["closed"]))
out(f"\nLargest gap between simulated mean and closed form: {abs(worst['sim_mean'] - worst['closed']):.3f} "
    f"at phi={worst['phi']}, rho={worst['rho']}.")
out("From rho = 0.5 to 2 single runs spread widely. The store behaves like an urn: its first records set an error")
out("rate that later copies reinforce, and the pull back to e* is weak when phi is high. The closed form is the")
out("long-run answer; a finite run can sit far from it.")
out.save(data)
