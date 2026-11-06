"""Sections 6-8 and 14 (point 3): six advantage estimators on the same task, same budget, three seeds.

Every method sees 5 prompts x 8 rollouts per step for 200 steps. The critic method also trains a second network of
the same size. pass@1 is exact (summed over all completions), so the only noise is from training itself.

    python labs/reinforcement-learning-from-first-principles/scripts/07_compare_estimators.py
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rlfp.report import Report  # noqa: E402
from rlfp.train import TrainConfig, first_step_reaching, train  # noqa: E402

METHODS = ["reinforce", "batch", "rloo", "grpo", "critic_gae", "ppo"]


def main() -> None:
    rep = Report("07_compare_estimators")
    rows, runs = [], {}
    for method in METHODS:
        finals, reach, ents = [], [], []
        for seed in range(3):
            out = train(TrainConfig(method=method, steps=200, lr=3e-3, eval_every=10, seed=seed))
            hist = out["history"]
            finals.append(hist[-1]["pass@1"])
            ents.append(hist[-1]["seq_entropy"])
            reach.append(first_step_reaching(hist, "pass@1", 0.5))
            runs[f"{method}/seed{seed}"] = [{"step": h["step"], "pass@1": h["pass@1"]} for h in hist]
        hit = [r for r in reach if r is not None]
        rows.append({"method": method, "final_pass@1_mean": statistics.mean(finals),
                     "final_pass@1_min": min(finals), "final_pass@1_max": max(finals),
                     "steps_to_0.5": (f"{statistics.mean(hit):.0f}" if hit else "never") + f" ({len(hit)}/3)",
                     "seq_entropy": statistics.mean(ents)})
        rep.print("  ", rows[-1])
    rep.print("\n5 prompts x G = 8, 200 steps, Adam lr 3e-3 (critic lr 1e-2), seeds 0-2. pass@1 = 0.6 means the "
              "three common targets are solved and the two rare ones are not.")
    rep.table(rows)
    rep.data["rows"] = rows
    rep.data["runs"] = runs
    rep.save()


if __name__ == "__main__":
    main()
