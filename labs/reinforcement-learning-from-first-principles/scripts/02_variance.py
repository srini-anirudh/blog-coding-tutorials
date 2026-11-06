"""Section 5: why the estimator is so noisy, measured three ways.

1. Most samples contribute nothing: the share of R = 0 rollouts, and the share of groups where all G rewards are
   equal (so a group baseline gives every sample zero weight), against 1 - p^G - (1-p)^G from the exact pass rate.
2. Variance grows with sequence length. The reward depends only on the first token, so extra tokens add score
   terms with zero mean and nonzero variance. The post predicts SNR ~ 1 / sqrt(T).
3. Error falls like 1 / sqrt(N) (shown in 01_policy_gradient.py): halving it costs 4x the rollouts.

    python labs/reinforcement-learning-from-first-principles/scripts/02_variance.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rlfp.estimators import error_stats, flat_grad, pg_loss  # noqa: E402
from rlfp.exact import exact_J  # noqa: E402
from rlfp.policy import TinyLM, prompt_batch, response_logits, sample, token_logprobs  # noqa: E402
from rlfp.report import Report  # noqa: E402
from rlfp.tasks import key_token_reward  # noqa: E402
from rlfp.train import make_policy, rollout, sum_task  # noqa: E402


def zero_contribution(rep: Report) -> None:
    task = sum_task()
    policy = make_policy(task, hidden=64, seed=0)
    p_exact = [exact_J(lp.detach(), r).item() for lp, r in zip(task.logps(policy), task.rewards)]
    gen = torch.Generator().manual_seed(0)
    G, n_batches = 8, 400
    zero, same = torch.zeros(5), torch.zeros(5)
    for _ in range(n_batches):
        _, _, R = rollout(policy, task, G, gen)
        g = R.view(5, G)
        zero += (g == 0).float().mean(1)
        same += (g.min(1).values == g.max(1).values).float()
    rows = []
    for i, target in enumerate(task.prompt_args):
        p = p_exact[i]
        rows.append({"target_sum": target, "exact_pass_rate": p, "share_R0": float(zero[i] / n_batches),
                     "groups_all_equal": float(same[i] / n_batches),
                     "predicted_all_equal": p ** G + (1 - p) ** G})
    rep.print(f"1. Untrained policy, G = {G}, {n_batches} groups per prompt")
    rep.table(rows)
    rep.data["zero_contribution"] = rows


def length_snr(rep: Report) -> None:
    n_actions, N, K = 6, 16, 300
    rep.print(f"\n2. Reward depends only on token 0. N = {N} rollouts per estimate, {K} estimates per T")
    rows = []
    for T in [2, 8, 32, 128]:
        torch.manual_seed(0)
        policy = TinyLM(n_actions + 1, n_actions, hidden=64)
        params = list(policy.parameters())
        prompts = prompt_batch([n_actions], 1)
        # Exact J = pi(a_0 = key | prompt), so its gradient is exact without listing 6^128 completions.
        p0 = torch.softmax(response_logits(policy, prompts, torch.zeros(1, 1, dtype=torch.long))[0, 0], -1)
        J = p0[2]
        g_true = flat_grad(J, params)
        gen = torch.Generator().manual_seed(T)
        ests = []
        for _ in range(K):
            pb = prompt_batch([n_actions], N)
            acts = sample(policy, pb, T, gen)
            loss = pg_loss(token_logprobs(policy, pb, acts), key_token_reward(acts, (0, 2)))
            ests.append(flat_grad(-loss, params))
        st = error_stats(torch.stack(ests), g_true)
        rows.append({"T": T, "pass_rate": J.item(), "snr": st["snr"], "snr_x_sqrtT": st["snr"] * math.sqrt(T),
                     "cos_to_true": st["cos"]})
    rep.table(rows)
    slope = (math.log(rows[-1]["snr"]) - math.log(rows[0]["snr"])) / (math.log(rows[-1]["T"]) - math.log(rows[0]["T"]))
    rep.print(f"log-log slope of SNR against T: {slope:.3f} (the post's 1/sqrt(T) rule predicts -0.5)")
    rep.data["length_snr"] = {"rows": rows, "slope": slope}


def main() -> None:
    rep = Report("02_variance")
    zero_contribution(rep)
    length_snr(rep)
    rep.save()


if __name__ == "__main__":
    main()
