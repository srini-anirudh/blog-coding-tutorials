"""SFT and RL as operators: SFT is a policy gradient with reward 1/pi; RL cannot reach what it never samples.

    python labs/how-reinforcement-learning-teaches-models-to-reason/scripts/09_sft_vs_rl.py
"""

import _path  # noqa: F401
import torch

from rlvrlab.common import Report
from rlvrlab.operators import one_step_ratios, run_operator, sft_grad_as_policy_gradient, sft_grad_exact, token_weights

K, TARGET = 12, 11


def start_logits() -> torch.Tensor:
    theta = torch.tensor([2.0, 1.5, 1.0, 1.0, 0.5, 0.0, 0.0, -0.5, -1.0, -1.0, -1.5, 0.0], dtype=torch.float64)
    theta[TARGET] = -12.0  # the demonstrated strategy: about 1e-6 under the starting policy
    return theta


def main() -> None:
    rep = Report("09_sft_vs_rl")
    theta = start_logits()
    pi = theta.softmax(-1)
    rep.print(f"{K} modes. Demonstration mode {TARGET} starts at pi = {float(pi[TARGET]):.2e}.")
    rep.print()

    rep.print("1. SFT gradient written as an on-policy policy gradient with reward D(y)/pi(y):")
    exact = sft_grad_exact(theta, TARGET)
    gen = torch.Generator().manual_seed(0)
    rows = []
    for n in (10**3, 10**5, 10**7):
        est = sft_grad_as_policy_gradient(theta, TARGET, n, gen)
        rows.append([n, float((est - exact).norm() / exact.norm())])
    rep.table(["on-policy samples", "relative error"], rows)
    mild = theta.clone()
    mild[TARGET] = 0.0
    est = sft_grad_as_policy_gradient(mild, TARGET, 10**5, gen)
    mild_err = float((est - sft_grad_exact(mild, TARGET)).norm() / sft_grad_exact(mild, TARGET).norm())
    rep.print(f"same check with the demonstration at pi = {float(mild.softmax(-1)[TARGET]):.3f}, 1e5 samples: "
              f"relative error {mild_err:.4f}")
    rep.print()

    rep.print("2. Per-token weights of a demonstration token with probability p:")
    p = torch.tensor([1.0, 0.5, 0.1, 0.01, 0.001], dtype=torch.float64)
    w = token_weights(p)
    rep.table(["p", "implicit reward SFT", "implicit reward DFT", "logit-grad norm SFT", "logit-grad norm DFT"],
              [[float(p[i])] + [float(w[k][i]) for k in w] for i in range(len(p))])
    rep.print()

    rep.print("3. One step of each operator: pi_new / pi_old for the non-demonstration modes.")
    rewarded = torch.zeros(K, dtype=torch.float64)
    rewarded[[2, 3, TARGET]] = 1.0  # modes 2 and 3 are correct and inside the support
    sft_ratio = one_step_ratios(theta, exact, lr=1.0)
    exact_rl = pi * (rewarded - (pi * rewarded).sum())  # the exact (expected) RL ascent direction
    rl_ratio = one_step_ratios(theta, exact_rl / exact_rl.norm() * exact.norm(), lr=1.0)
    others = [i for i in range(K) if i != TARGET]
    rep.table(["mode", "correct?", "SFT ratio", "RL ratio"],
              [[i, int(rewarded[i]), float(sft_ratio[i]), float(rl_ratio[i])] for i in others])
    rep.print("SFT's change to each other mode depends only on that mode's probability, not on whether it is right.")
    rep.print("The modes it lowers are the most likely ones, through normalization. RL raises the correct modes")
    rep.print("and lowers every wrong one. Both steps have the same logit-space length.")
    rep.print()

    rep.print("4. 300 steps of each, lr 1.0. RL: GRPO with 8 samples per step, reward 1 on modes 2, 3 and the target.")
    sft_theta, _ = run_operator(theta, "sft", TARGET, rewarded, 300, 1.0)
    rl_theta, seen = run_operator(theta, "rl", TARGET, rewarded, 300, 1.0)
    rows = []
    for name, t in (("start", theta), ("after SFT", sft_theta), ("after RL", rl_theta)):
        q = t.softmax(-1)
        rows.append([name, float(q[TARGET]), float((q * rewarded).sum()), float(-(q * q.log()).sum())])
    rep.table(["policy", "pi(target)", "P(correct)", "entropy"], rows)
    rep.print(f"RL sampled the target mode {seen} times in {300 * 8} samples.")
    rep.data = {"identity_check": rows, "mild_err": mild_err, "target_seen_by_rl": seen,
                "sft_ratio": sft_ratio.tolist(), "rl_ratio": rl_ratio.tolist()}
    rep.save()


if __name__ == "__main__":
    main()
