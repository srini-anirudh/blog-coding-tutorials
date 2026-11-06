"""Section 6: baselines, and why they are free.

1. E[b grad log pi] = 0. Summed exactly over every completion, the baseline term vanishes to float precision.
2. The variance parabola. For one sample, E||(R - b) s||^2 - ||grad J||^2 is a quadratic in b, where s is the score
   grad log pi(tau). It is computed exactly from every completion's score norm. The post says the minimum sits at
   the mean reward when score size does not depend on reward. The exact minimiser is E[R ||s||^2] / E[||s||^2].
3. The self-inclusion bias. A batch mean that includes the sample being weighted scales the gradient by (N-1)/N.
   Leave-one-out removes it.
4. A per-state baseline b(s_t) = V(s_t), from the exact values, against constant baselines.

    python labs/reinforcement-learning-from-first-principles/scripts/03_baselines.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rlfp.estimators import baseline_weights, error_stats, flat_grad, pg_loss  # noqa: E402
from rlfp.exact import ExactTask, exact_J, prefix_values, seq_logprobs  # noqa: E402
from rlfp.policy import prompt_batch, sample, token_logprobs  # noqa: E402
from rlfp.report import Report  # noqa: E402
from rlfp.tasks import sum_reward  # noqa: E402
from rlfp.train import make_policy, sum_task  # noqa: E402


def exact_parts(policy, pid, seqs, R):
    params = list(policy.parameters())
    logp = seq_logprobs(policy, pid, seqs)
    g_true = flat_grad(exact_J(logp, R), params, retain_graph=True)
    g_base = flat_grad((logp.detach().exp() * logp).sum(), params, retain_graph=True)  # E[1 * grad log pi]
    return logp, params, g_true, g_base


def main() -> None:
    rep = Report("03_baselines")
    task = sum_task()
    policy = make_policy(task, hidden=64, seed=0)
    pid, target = task.prompt_ids[2], task.prompt_args[2]
    R = task.reward_fn(task.seqs, target)
    logp, params, g_true, g_base = exact_parts(policy, pid, task.seqs, R)
    rep.print(f"1. ||E[grad log pi]|| = {float(g_base.norm()):.2e}   ||grad J|| = {float(g_true.norm()):.2e}"
              f"   (sum over all {len(task.seqs)} completions)")
    rep.data["score_mean_norm"] = float(g_base.norm())
    rep.data["grad_norm"] = float(g_true.norm())

    # 2. The exact one-sample variance as a function of b, on a smaller task so every score can be listed.
    small = ExactTask(4, 4, sum_reward, [4], [6])
    sp = make_policy(small, hidden=32, seed=0)
    sR = small.rewards[0].double()
    sparams = list(sp.parameters())
    slogp = seq_logprobs(sp, 4, small.seqs)
    p = slogp.detach().exp().double()
    s2 = torch.stack([flat_grad(slogp[i], sparams, retain_graph=True).double().pow(2).sum()
                      for i in range(len(small.seqs))])
    g2 = float(flat_grad(exact_J(slogp, small.rewards[0]), sparams).double().pow(2).sum())
    a, c, d = float((p * s2).sum()), float((p * sR * s2).sum()), float((p * sR * sR * s2).sum())
    mean_R, b_star = float((p * sR).sum()), c / a

    def var(b: float) -> float:
        return a * b * b - 2 * c * b + d - g2

    rep.print(f"\n2. One-sample variance tr Cov[(R - b) grad log pi] on 4 digits x 4 tokens, sum = 6 "
              f"({len(small.seqs)} completions)")
    rows = [{"b": b, "variance": var(b), "vs_b0": var(b) / var(0.0)}
            for b in [-0.5, 0.0, mean_R / 2, mean_R, b_star, 0.5, 1.0]]
    rep.table(rows)
    rep.print(f"mean reward E[R] = {mean_R:.4f}; exact minimiser E[R|s|^2]/E[|s|^2] = {b_star:.4f}; "
              f"corr(R, |s|^2) under pi = {corr(p, sR, s2):.3f}")
    rep.data["parabola"] = {"rows": rows, "mean_R": mean_R, "b_star": b_star}

    # 3 and 4: sampled estimates, N = 4 per estimate, against the exact gradient.
    values = prefix_values(logp, R, task.seqs, task.n_actions)  # (S, T + 1)
    code_of = {tuple(s.tolist()): i for i, s in enumerate(task.seqs)}
    gen = torch.Generator().manual_seed(0)
    N, K = 4, 3000
    ests = {k: [] for k in ["none", "batch", "loo", "state_value"]}
    for _ in range(K):
        pb = prompt_batch([pid], N)
        acts = sample(policy, pb, task.T, gen)
        r = task.reward_fn(acts, target)
        tlp = token_logprobs(policy, pb, acts)
        idx = torch.tensor([code_of[tuple(a.tolist())] for a in acts])
        for kind in ests:
            w = (r.unsqueeze(1) - values[idx, :-1].float()) if kind == "state_value" else baseline_weights(r, kind)
            ests[kind].append(flat_grad(-pg_loss(tlp, w), params, retain_graph=True))
    rows = []
    for kind, e in ests.items():
        e = torch.stack(e)
        st = error_stats(e, g_true)
        m = e.mean(0)
        proj = float(m @ g_true / g_true.pow(2).sum())
        se = float((e @ g_true / g_true.pow(2).sum()).std() / K ** 0.5)
        rows.append({"baseline": kind, "mean_projection": proj, "std_error": se, "mse_vs_none": st["mse"]})
    base = rows[0]["mse_vs_none"]
    for row in rows:
        row["mse_vs_none"] = row["mse_vs_none"] / base
    rep.print(f"\n3-4. Sum = {target} task, N = {N} samples per estimate, {K} estimates. mean_projection is "
              f"<mean estimate, grad J> / ||grad J||^2: 1 means unbiased, (N-1)/N = {(N - 1) / N} is the predicted "
              f"self-inclusion bias.")
    rep.table(rows)
    rep.data["sampled"] = rows
    rep.save()


def corr(p, x, y) -> float:
    mx, my = (p * x).sum(), (p * y).sum()
    cov = (p * (x - mx) * (y - my)).sum()
    return float(cov / ((p * (x - mx) ** 2).sum().sqrt() * (p * (y - my) ** 2).sum().sqrt()))


if __name__ == "__main__":
    main()
