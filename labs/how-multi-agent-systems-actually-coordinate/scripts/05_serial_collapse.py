"""Serial collapse and the PARL reward terms, in a toy orchestrator bandit (not Kimi's system).

    python labs/how-multi-agent-systems-actually-coordinate/scripts/05_serial_collapse.py [--seeds 20]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mascoord import parl as P  # noqa: E402
from mascoord.report import Report  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=20)
ap.add_argument("--steps", type=int, default=1500)
args = ap.parse_args()
out = Report("05_serial_collapse")

out("\nTrue objective r_perf = success - 0.3 x wall clock, for each choice of k subagents")
out(f"{'k':>4} {'skill 0':>8} {'skill 1':>8}")
for k, a, b in zip(P.ACTIONS, P.success_prob(P.ACTIONS, 0) - P.TIME_WEIGHT * P.wall_clock(P.ACTIONS),
                   P.success_prob(P.ACTIONS, 1) - P.TIME_WEIGHT * P.wall_clock(P.ACTIONS)):
    out(f"{k:>4} {a:>8.3f} {b:>8.3f}")
out("   Serial (k=0) is best until the decomposition skill is learned; k=4 is best after.")

out(f"\nREINFORCE, {args.steps} steps x 32 episodes, {args.seeds} seeds, initial P(serial) = "
    f"{np.exp(P.SERIAL_PRIOR) / (np.exp(P.SERIAL_PRIOR) + 4):.2f}")
out(f"{'reward':<38} {'final policy over k = 0,1,2,4,8':<34} {'skill':>6} {'collapsed':>10} "
    f"{'spurious/ep':>12} {'true r_perf':>12}")
summary = {}
for c in P.CONDITIONS:
    runs = [P.train(c, steps=args.steps, seed=s) for s in range(args.seeds)]
    probs = np.mean([r["probs"] for r in runs], 0)
    skill = np.mean([r["skill"] for r in runs])
    collapsed = sum(r["probs"][0] > 0.5 for r in runs)
    spurious = np.mean([np.mean(r["trace"]["spurious"]) for r in runs])
    true = np.mean([P.expected_r_perf(r["probs"], r["skill"]) for r in runs])
    out(f"{c.name:<38} {str(np.round(probs, 2)):<34} {skill:>6.2f} {collapsed:>5}/{args.seeds:<4} "
        f"{spurious:>12.2f} {true:>12.3f}")
    summary[c.name] = {"probs": probs.round(3).tolist(), "collapsed": int(collapsed), "true_r_perf": round(true, 3)}
out("   collapsed = seeds whose final P(serial) > 0.5. spurious/ep = subagents beyond the 4 real")
out("   subtasks, averaged over all training episodes.")

out("\nHow much the collapse depends on the starting preference (r_perf only)")
for prior in [0.0, 1.0, 2.0, 3.0]:
    runs = [P.train(P.CONDITIONS[0], steps=args.steps, serial_prior=prior, seed=s) for s in range(args.seeds)]
    p0 = np.exp(prior) / (np.exp(prior) + 4)
    out(f"   initial P(serial) = {p0:.2f}: collapsed in {sum(r['probs'][0] > 0.5 for r in runs)}/{args.seeds} seeds")
out.json({"conditions": summary})
