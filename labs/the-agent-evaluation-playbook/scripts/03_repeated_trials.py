"""pass^k is a variance meter, and accuracy is half of a number.

Four task suites with the same mean success and different spread run through the harness 16 times per task.
The per-task pass^k estimates are compared with E[p^k], pbar^k and the second-order formula. The same runs then
price pass@k: retry loops buy accuracy with calls, and U = Q - lambda C picks a different winner as lambda grows.

    python labs/the-agent-evaluation-playbook/scripts/03_repeated_trials.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agenteval.agents import noisy  # noqa: E402
from agenteval.graders import grade_state  # noqa: E402
from agenteval.harness import run_episode  # noqa: E402
from agenteval.reliability import (difficulty_suites, pass_at_k, pass_hat_k, second_order, suite_curve,  # noqa: E402
                                   utility)
from agenteval.report import Report  # noqa: E402
from agenteval.world import cancel_and_refund, initial_state  # noqa: E402

N_TASKS, N_TRIALS, MEAN, KS = 200, 16, 0.6, (1, 2, 4, 8)


def run_suite(ps, rng):
    """Return success[task, trial], cost[task, trial] and outcome counts, from real harness episodes."""
    task, s0 = cancel_and_refund("A100"), initial_state()
    succ = np.zeros((len(ps), N_TRIALS), bool)
    cost = np.zeros((len(ps), N_TRIALS))
    outcomes = {"success": 0, "wrong end state": 0, "budget exhausted": 0}
    for i, p in enumerate(ps):
        agent = noisy(float(p))
        for t in range(N_TRIALS):
            ep = run_episode(agent, task, s0, rng=rng, max_calls=12, episode=t)
            succ[i, t], cost[i, t] = grade_state(ep), ep.cost
            outcomes["success" if succ[i, t] else
                     "budget exhausted" if ep.outcome == "budget_exhausted" else "wrong end state"] += 1
    return succ, cost, outcomes


def main() -> None:
    r = Report("03_repeated_trials")
    rng = np.random.default_rng(0)
    suites = difficulty_suites(N_TASKS, MEAN, rng)
    r.add(f"{N_TASKS} tasks per suite, {N_TRIALS} harness episodes per task, scripted noisy agent whose per-task")
    r.add(f"success probability p_i is drawn from the suite with mean {MEAN}. Each sampled mean is printed below.")
    r.add()
    runs = {}
    for name, ps in suites.items():
        succ, cost, outcomes = run_suite(ps, rng)
        runs[name] = (ps, succ, cost)
        c = succ.sum(1)
        meas = suite_curve(c, N_TRIALS, KS)
        rows = [[f"k={k}", meas[j], float(np.mean(ps ** k)), ps.mean() ** k, second_order(ps.mean(), ps.var(), k)]
                for j, k in enumerate(KS)]
        r.add(f"suite: {name}   mean p = {ps.mean():.3f}, Var(p) = {ps.var():.4f}   "
              f"outcomes: {outcomes}")
        r.table(["", "measured pass^k", "E[p^k] (true p_i)", "pbar^k", "2nd-order approx"], rows)

    r.add("pass@k against pass^k on the same runs (measured)")
    rows = []
    for name, (ps, succ, _) in runs.items():
        c = succ.sum(1)
        rows.append([name] + [suite_curve(c, N_TRIALS, [k], pass_at_k)[0] for k in KS] +
                    [suite_curve(c, N_TRIALS, [k], pass_hat_k)[0] for k in KS])
    r.table(["suite"] + [f"pass@{k}" for k in KS] + [f"pass^{k}" for k in KS], rows)

    r.add("Cost (beta, wide suite). Each tool call is priced at $0.01, an assumed price. A retry loop uses the")
    r.add("goal-state check as its verifier, so it stops at the first success. 'best-of-8' always runs 8 attempts.")
    r.add("Q for a config that allows k attempts uses the first k of each task's 16 episodes.")
    ps, succ, cost = runs["beta, wide"]
    configs = {}
    for k in (1, 3, 8):
        first = np.where(succ[:, :k].any(1), succ[:, :k].argmax(1), k - 1)
        spent = np.array([cost[i, :first[i] + 1].sum() for i in range(len(ps))])
        configs[f"retry up to {k}" if k > 1 else "single attempt"] = (succ[:, :k].any(1).mean(), spent.mean())
    configs["best-of-8"] = (succ[:, :8].any(1).mean(), cost[:, :8].sum(1).mean())
    fail_cost = cost[:, 0][~succ[:, 0]].mean()
    win_cost = cost[:, 0][succ[:, 0]].mean()
    lams = (0.0, 1.0, 5.0, 20.0)
    rows = [[name, q, c] + [utility(q, c, lam) for lam in lams] for name, (q, c) in configs.items()]
    r.table(["config", "Q (solved)", "C ($/task)"] + [f"U, lambda={lam:g}" for lam in lams], rows)
    for lam in lams:
        best = max(configs, key=lambda n: utility(*configs[n], lam))
        r.add(f"   lambda = {lam:g} per dollar: best config is {best}")
    r.add(f"   cost of one attempt: ${win_cost:.3f} when it succeeds, ${fail_cost:.3f} when it fails "
          f"(loops run to the 12-call budget)")
    r.save()


if __name__ == "__main__":
    main()
