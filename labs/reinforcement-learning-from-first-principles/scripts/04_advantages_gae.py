"""Sections 7 and 8: values, advantages, TD errors and the GAE lambda dial.

1. Exact V(s_t), Q(s_t, a_t) and A = Q - V for every prefix of every completion. Credit is not flat: on a task where
   only token 3 matters, |A| is zero everywhere else. E_a[A(s, a)] = 0 at every state.
2. With gamma = lambda = 1 and a terminal reward, GAE collapses to R - V(s_t).
3. The lambda dial. The critic is the exact V plus noise of size sigma. "Redrawn" noise is fresh for every visit.
   "Persistent" noise is a fixed error per state, like a real critic's. The error of the resulting policy gradient
   is measured against the exact gradient, at N = 64 and N = 1024 rollouts per estimate.

    python labs/reinforcement-learning-from-first-principles/scripts/04_advantages_gae.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rlfp.advantages import gae, terminal_rewards  # noqa: E402
from rlfp.estimators import error_stats, flat_grad, pg_loss  # noqa: E402
from rlfp.exact import ExactTask, exact_advantages, exact_J, prefix_values, seq_logprobs  # noqa: E402
from rlfp.policy import prompt_batch, sample, token_logprobs  # noqa: E402
from rlfp.report import Report  # noqa: E402
from rlfp.tasks import key_token_reward  # noqa: E402
from rlfp.train import make_policy, sum_task  # noqa: E402


def credit_by_position(rep: Report) -> None:
    rep.print("1. Mean |A(s_t, a_t)| under pi, by position t, exact")
    rows = []
    key = ExactTask(6, 5, key_token_reward, [6], [(3, 2)])
    for name, task, i in [("key token at t=3", key, 0), ("digits sum to 12", sum_task(), 2)]:
        policy = make_policy(task, hidden=64, seed=0)
        with torch.no_grad():
            logp = seq_logprobs(policy, task.prompt_ids[i], task.seqs)
        V = prefix_values(logp, task.rewards[i], task.seqs, task.n_actions)
        A = exact_advantages(V)
        p = logp.exp().double()
        row = {"task": name}
        for t in range(task.T):
            row[f"t={t}"] = float((p * A[:, t].abs()).sum())
        # E_a[A] = 0 at every state: the p-weighted A, summed within each prefix, is zero.
        worst = 0.0
        code = torch.zeros(len(task.seqs), dtype=torch.long)
        for t in range(task.T):
            num = torch.zeros(task.n_actions ** t, dtype=torch.float64).index_add_(0, code, p * A[:, t])
            worst = max(worst, float(num.abs().max()))
            code = code * task.n_actions + task.seqs[:, t]
        row["max|E_a[A]|"] = worst
        rows.append(row)
    rep.table(rows)
    rep.data["credit_by_position"] = rows


def gae_dial(rep: Report) -> None:
    task = sum_task()
    i = 2
    policy = make_policy(task, hidden=64, seed=0)
    params = list(policy.parameters())
    pid, target = task.prompt_ids[i], task.prompt_args[i]
    logp = seq_logprobs(policy, pid, task.seqs)
    g_true = flat_grad(exact_J(logp, task.rewards[i]), params)
    V = prefix_values(logp, task.rewards[i], task.seqs, task.n_actions).float()  # (S, T+1)
    weights = torch.tensor([6 ** (task.T - 1 - t) for t in range(task.T)])
    gen = torch.Generator().manual_seed(0)

    # Persistent critic error: one fixed draw per prefix (state), shared by every rollout that visits it.
    n_states = sum(6 ** t for t in range(task.T))
    fixed = torch.randn(n_states, generator=torch.Generator().manual_seed(7))

    def state_ids(acts):  # unique id of the prefix a_<t, for t = 0..T-1
        ids, offset, code = [], 0, torch.zeros(len(acts), dtype=torch.long)
        for t in range(task.T):
            ids.append(offset + code)
            offset += 6 ** t
            code = code * 6 + acts[:, t]
        return torch.stack(ids, 1)

    # 2. The collapse at gamma = lambda = 1.
    acts = sample(policy, prompt_batch([pid], 256), task.T, gen)
    idx = (acts * weights).sum(1)
    R = task.reward_fn(acts, target)
    v = V[idx, :-1]
    collapse = float((gae(terminal_rewards(R, task.T), v, 1.0, 1.0) - (R.unsqueeze(1) - v)).abs().max())
    rep.print(f"\n2. gamma = lambda = 1: max |GAE - (R - V(s_t))| = {collapse:.2e}")
    rep.data["gae_collapse_max_diff"] = collapse

    rep.print("\n3. Relative MSE of the policy gradient, ||g - grad J||^2 / ||grad J||^2, with GAE advantages")
    rows = []
    lams = [0.0, 0.5, 0.8, 0.95, 1.0]
    settings = [("exact", 0.0, 64), ("redrawn", 0.08, 64), ("persistent", 0.08, 64), ("persistent", 0.08, 1024),
                ("redrawn", 0.2, 64), ("persistent", 0.2, 64), ("persistent", 0.2, 1024)]
    K = 60
    for mode, sigma, N in settings:
        row = {"critic": mode, "sigma": sigma, "N": N}
        for lam in lams:
            g_ests = []
            gen = torch.Generator().manual_seed(1)
            for _ in range(K):
                pb = prompt_batch([pid], N)
                acts = sample(policy, pb, task.T, gen)
                idx = (acts * weights).sum(1)
                R = task.reward_fn(acts, target)
                v = V[idx, :-1].clone()
                if mode == "redrawn":
                    v += sigma * torch.randn(v.shape, generator=gen)
                elif mode == "persistent":
                    v += sigma * fixed[state_ids(acts)]
                adv = gae(terminal_rewards(R, task.T), v, 1.0, lam)
                g_ests.append(flat_grad(-pg_loss(token_logprobs(policy, pb, acts), adv), params))
            st = error_stats(torch.stack(g_ests), g_true)
            row[f"lam={lam}"] = st["mse"] / st["true_norm_sq"]
        rows.append(row)
        rep.print("  ", row)
    rep.table(rows)
    rep.data["gae_dial"] = rows


def main() -> None:
    rep = Report("04_advantages_gae")
    credit_by_position(rep)
    gae_dial(rep)
    rep.save()


if __name__ == "__main__":
    main()
