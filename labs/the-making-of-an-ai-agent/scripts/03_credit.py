"""Section "Credit assignment over long horizons": GRPO's episode-level advantage vs GiGPO's anchor-state groups.

    python labs/the-making-of-an-ai-agent/scripts/03_credit.py            # 5 seeds x 4 settings, about 2 minutes
"""

import argparse
import random
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from agentcraft.common import table, write_results
from agentcraft.credit import (ChainConfig, anchor_advantages, episode_advantages, make_roles, run_episode,
                               step_returns, train)

SETTINGS = [("grpo", 0.0), ("gigpo", 0.0), ("gigpo", 0.5), ("gigpo", 1.0)]


def job(args):
    method, noise, seed, iters = args
    torch.set_num_threads(1)
    return train(method, iters, seed, state_noise=noise, log=lambda s: print(s, flush=True))


def illustrate(cfg: ChainConfig) -> str:
    """Four rollouts of one task from the untrained (uniform) policy, with both advantages written per step."""
    roles = make_roles(cfg, seed=0)
    probs = torch.full((cfg.n_tasks, cfg.stages, cfg.actions), 1 / cfg.actions)
    for seed in range(1000):
        rng = random.Random(seed)
        group = [run_episode(0, probs, roles, cfg, rng) for _ in range(4)]
        if sum(e.reward for e in group) == 1 and all(len(e.actions) > 2 for e in group):
            break
    a_e = episode_advantages(group)
    a_s, _ = anchor_advantages(group, cfg.gamma)
    lines = [f"Task 0, 4 rollouts of the untrained policy (rollout seed {seed}). Each step: stage:role "
             "A_E / A_S.", ""]
    for i, e in enumerate(group):
        R = step_returns(e, cfg.gamma)
        steps = [f"s{s}:{roles[0][s][a]} {a_e[i]:+.2f}/{a_s[i][t]:+.2f} (R_t {R[t]:.2f})"
                 for t, (s, a) in enumerate(zip(e.states, e.actions))]
        lines.append(f"rollout {i + 1}, R = {e.reward:.0f}:  " + " | ".join(steps))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    cfg = ChainConfig()
    jobs = [(m, n, s, a.iters) for m, n in SETTINGS for s in range(a.seeds)]
    with ProcessPoolExecutor(a.workers) as pool:
        runs = list(pool.map(job, jobs))

    rows = []
    iters = [h["iter"] for h in runs[0]["history"]]
    for m, n in SETTINGS:
        rs = [r for r in runs if r["method"] == m and r["state_noise"] == n]
        for i, it in enumerate(iters):
            succ = [r["history"][i]["success"] for r in rs]
            noop = [r["history"][i]["noop_prob"] for r in rs]
            rows.append([f"{m}" + (f", state noise {n}" if m == "gigpo" else ""), it,
                         f"{statistics.mean(succ):.3f} +- {statistics.stdev(succ):.3f}",
                         f"{statistics.mean(noop):.3f} +- {statistics.stdev(noop):.3f}",
                         statistics.mean(r["history"][i]["anchor_coverage"] for r in rs)])
    text = (f"{cfg.n_tasks} tasks, {cfg.stages} stages, {cfg.actions} actions per stage (1 advance, 1 dead end, "
            f"{cfg.actions - 2} no-ops), horizon {cfg.horizon}, gamma {cfg.gamma}.\nTabular softmax policy, Adam lr "
            f"0.1, 16 tasks x 8 rollouts per iteration, omega 1. {a.seeds} seeds, mean +- std.\n"
            "success = exact P(success) averaged over tasks (dynamic programming, no sampling noise).\n"
            "noop prob = probability mass the policy puts on no-op actions, averaged over states.\n"
            "anchor coverage = share of steps whose anchor group has 2 or more members (seen by GiGPO's matcher).\n\n"
            + table(rows, ["method", "iteration", "success", "noop prob", "anchor coverage"])
            + "\n\n" + illustrate(cfg))
    print(text)
    write_results("03_credit", text, {"config": vars(cfg), "runs": runs})


if __name__ == "__main__":
    main()
