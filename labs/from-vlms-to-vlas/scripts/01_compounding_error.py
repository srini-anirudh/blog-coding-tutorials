"""Compounding error in imitation, and its two remedies: fewer decisions (chunks) and recovery data (DAgger).

Part A simulates the Ross and Bagnell argument exactly as the post states it. Part B trains real policies on a lane
task: behaviour cloning from clean expert demos, then DAgger, which adds expert labels on the learner's own states.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vlalab.envs import LaneTask  # noqa: E402
from vlalab.heads import RegressionHead, train  # noqa: E402
from vlalab.report import Report  # noqa: E402


def worst_case_mistakes(eps: float, T: int, H: int, n: int, rng) -> float:
    """One decision every H steps. Each decision on demo states is wrong with probability eps. After the first wrong
    decision the policy is off the demos and every later step is a mistake."""
    decisions = -(-T // H)
    first = rng.geometric(eps, n)  # index of the first wrong decision, 1-based
    start_step = (first - 1) * H  # 0-based step where the wrong chunk begins
    return float(np.where(first <= decisions, T - start_step, 0).mean())


def exact_mistakes(eps: float, T: int, H: int) -> float:
    decisions = -(-T // H)
    j = np.arange(1, decisions + 1)
    return float((eps * (1 - eps) ** (j - 1) * (T - (j - 1) * H)).sum())


def part_a(rep: Report, rng):
    eps = 0.01
    rep.line(f"Part A. Ross-Bagnell worst case, eps = {eps} per decision, 200000 Monte Carlo episodes.")
    rep.line("T = steps per task. H = actions per decision (chunk size). eps*T is the supervised-learning count.")
    rows = []
    for T in [10, 50, 100, 500]:
        for H in [1, 10, 50]:
            if H > T:
                continue
            mc = worst_case_mistakes(eps, T, H, 200_000, rng)
            rows.append({"T": T, "H": H, "decisions": -(-T // H), "mistakes_mc": round(mc, 2),
                         "mistakes_exact": round(exact_mistakes(eps, T, H), 2), "eps_T": eps * T,
                         "eps_T2_over_2H": round(eps * T * T / (2 * H), 2)})
    rep.table(rows)
    rep.line("At T = 500 (ten seconds at 50 Hz), one-step decisions give about "
             f"{[r for r in rows if r['T'] == 500 and r['H'] == 1][0]['mistakes_mc']} mistaken steps; "
             f"one-second chunks (H = 50) give {[r for r in rows if r['T'] == 500 and r['H'] == 50][0]['mistakes_mc']}.")
    return rows


def lane_policy(head, task):
    def f(t, y):
        with torch.no_grad():
            return head.net(torch.as_tensor(task.features(t, y))).squeeze(-1).numpy().astype(np.float64)
    return f


def part_b(rep: Report, steps: int, seed: int):
    task, rng = LaneTask(), np.random.default_rng(seed)
    test_noise = dict(y0_std=0.02, act_noise=0.3)  # the same small disturbances for every policy
    # Clean demos: 50 episodes of 400 steps that start on the lane centre. The expert never leaves it.
    t, y = task.rollout(task.expert, 50, 400, rng, y0_std=0.002)
    data_obs, data_act = [task.features(t.ravel(), y.ravel())], [task.expert(t.ravel(), y.ravel())[:, None]]
    rep.line(f"\nPart B. Lane task, {task.__class__.__doc__.splitlines()[0]}")
    rep.line(f"Demos: 50 x 400 expert steps, |y| up to {np.abs(y).max():.4f}. Test: start offset std "
             f"{test_noise['y0_std']}, action noise std {test_noise['act_noise']}. Mistake: |y| > {task.band}.")
    results = {}
    horizons = [25, 50, 100, 200, 400]

    def evaluate(name, policy):
        row = {"policy": name}
        for T in horizons:
            _, ys = task.rollout(policy, 500, T, np.random.default_rng(1234), **test_noise)
            row[f"T={T}"] = round(float((np.abs(ys) > task.band).sum(1).mean()), 2)
        row["mean |y| at T=400"] = round(float(np.abs(ys).mean()), 3)
        results[name] = row
        return row

    rows = [evaluate("expert", task.expert)]
    for it in range(5):
        head = RegressionHead(5, 1, width=64)
        train(head, np.concatenate(data_obs), np.concatenate(data_act).astype(np.float32), steps=steps, seed=seed)
        pol = lane_policy(head, task)
        name = "behaviour cloning" if it == 0 else f"DAgger round {it}"
        if it in (0, 1, 4):
            rows.append(evaluate(name, pol))
        # DAgger: run the learner, ask the expert what it should have done on the states it reached.
        t, y = task.rollout(pol, 50, 400, rng, **test_noise)
        data_obs.append(task.features(t.ravel(), y.ravel()))
        data_act.append(task.expert(t.ravel(), y.ravel())[:, None])
    rep.line("Mean number of mistaken steps per episode, 500 test episodes:")
    rep.table(rows)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000, help="training steps per policy")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "results" / "01_compounding_error"))
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rep = Report(args.out, " ".join(["scripts/01_compounding_error.py"] + sys.argv[1:]))
    a = part_a(rep, np.random.default_rng(args.seed))
    b = part_b(rep, args.steps, args.seed)
    rep.save_json({"part_a": a, "part_b": b})


if __name__ == "__main__":
    main()
