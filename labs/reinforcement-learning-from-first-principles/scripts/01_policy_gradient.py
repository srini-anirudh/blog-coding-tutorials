"""Sections 3 and 4: the objective, and the policy gradient checked against the exact gradient.

1. J(theta) = sum_tau pi(tau) R(tau) is computed exactly over all 7,776 completions. Autograd through that sum gives
   the true gradient. The score-function form E[R grad log pi] is then computed exactly too, by weighting every
   completion's score by its probability. The two must agree to float precision.
2. REINFORCE averages N sampled scores. Its error falls like 1/sqrt(N), and the error times sqrt(N) stays flat.
3. The greedy trap: two policies with the same J. One is diverse, one has collapsed onto fixed answers. Only the
   diverse one still has a gradient.

    python labs/reinforcement-learning-from-first-principles/scripts/01_policy_gradient.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rlfp.estimators import flat_grad, pg_loss  # noqa: E402
from rlfp.exact import exact_J, seq_logprobs  # noqa: E402
from rlfp.policy import prompt_batch, sample, token_logprobs  # noqa: E402
from rlfp.report import Report  # noqa: E402
from rlfp.train import make_policy, sum_task  # noqa: E402


def finite_difference(policy, pid, seqs, R, seed: int, h: float = 1e-4) -> dict:
    """(J(theta + h v) - J(theta - h v)) / 2h against grad J . v, in float64 on a copy of the policy."""
    import copy

    pol = copy.deepcopy(policy).double()
    params = list(pol.parameters())
    gen = torch.Generator().manual_seed(seed)
    v = [torch.randn(p.shape, generator=gen, dtype=torch.float64) for p in params]
    norm = torch.sqrt(sum((x ** 2).sum() for x in v))
    v = [x / norm for x in v]
    Rd = R.double()
    logp = seq_logprobs(pol, pid, seqs)
    g = flat_grad((logp.detach().exp() * Rd * logp).sum(), params)  # the score-function form
    analytic = float(g @ torch.cat([x.reshape(-1) for x in v]))
    with torch.no_grad():
        vals = []
        for sign in (1, -1):
            for p, x in zip(params, v):
                p.add_(sign * h * x)
            vals.append(float(exact_J(seq_logprobs(pol, pid, seqs), Rd)))
            for p, x in zip(params, v):
                p.sub_(sign * h * x)
    return {"fd": (vals[0] - vals[1]) / (2 * h), "analytic": analytic}


def main() -> None:
    rep = Report("01_policy_gradient")
    task = sum_task()
    policy = make_policy(task, hidden=64, seed=0)
    params = list(policy.parameters())
    pid, target = task.prompt_ids[2], task.prompt_args[2]
    R = task.reward_fn(task.seqs, target)
    rep.print(f"task: 6 digits, 5 tokens, reward 1 if the digits sum to {target}; {len(task.seqs)} completions")

    logp = seq_logprobs(policy, pid, task.seqs)
    J = exact_J(logp, R)
    g_true = flat_grad(J, params, retain_graph=True)
    # Score-function form, summed exactly: sum_tau pi(tau) R(tau) grad log pi(tau).
    g_score = flat_grad((logp.detach().exp() * R * logp).sum(), params)
    rel = float((g_score - g_true).norm() / g_true.norm())
    J = J.item()
    rep.print(f"J = {J:.4f}   ||grad J|| = {float(g_true.norm()):.4e}   parameters = {g_true.numel()}")
    rep.print(f"exact E[R grad log pi] vs autograd of the exact sum: relative difference {rel:.2e}")
    # An independent check that does not use autograd at all: a central finite difference along a random direction.
    fd = finite_difference(policy, pid, task.seqs, R, seed=3)
    rep.print(f"finite difference along a random direction (float64): {fd['fd']:.6e}   "
              f"score-function gradient . direction: {fd['analytic']:.6e}")
    rep.data.update(J=J, grad_norm=float(g_true.norm()), score_identity_rel_diff=rel, finite_difference=fd)

    rep.print("\nREINFORCE with N samples. RMS error relative to ||grad J||; unbiased means error x sqrt(N) is flat")
    gen = torch.Generator().manual_seed(1)
    rows = []
    for N in [1, 4, 16, 64, 256, 1024]:
        ests = []
        for _ in range(400 if N <= 16 else 50):
            prompts = prompt_batch([pid], N)
            acts = sample(policy, prompts, task.T, gen)
            loss = pg_loss(token_logprobs(policy, prompts, acts), task.reward_fn(acts, target))
            ests.append(flat_grad(-loss, params))  # pg_loss is minus the objective
        ests = torch.stack(ests)
        rel_err = float((ests - g_true).pow(2).sum(1).mean().sqrt() / g_true.norm())  # RMS error
        cos = float(((ests @ g_true) / (ests.norm(dim=1).clamp(min=1e-30) * g_true.norm())).mean())
        rows.append({"N": N, "estimates": len(ests), "rms_rel_error": rel_err, "rms_rel_error_x_sqrtN": rel_err * N ** 0.5,
                     "cos_to_true": cos, "zero_frac": float((ests.norm(dim=1) == 0).float().mean())})
    rep.table(rows)
    rep.data["reinforce_vs_N"] = rows
    rep.print("zero_frac is the share of estimates where every sample scored 0, so the estimate is the zero vector.")

    rep.print("\nThe greedy trap: 5 prompts, 7 answers each, answers 0-2 correct. Both policies have J = 0.6.")
    rewards = torch.tensor([1.0, 1, 1, 0, 0, 0, 0])
    diverse = torch.zeros(5, 7, requires_grad=True)  # filled below: P(correct) = 0.6 on every prompt
    with torch.no_grad():
        diverse[:, :3] = torch.log(torch.tensor(0.6 / 3))
        diverse[:, 3:] = torch.log(torch.tensor(0.4 / 4))
    collapsed = torch.full((5, 7), -30.0)
    collapsed[[0, 1, 2], [0, 1, 2]] = 0.0  # prompts 0-2 always give a correct answer
    collapsed[[3, 4], [3, 4]] = 0.0  # prompts 3-4 always give the same wrong answer
    collapsed.requires_grad_(True)
    trap = []
    for name, logits in [("diverse", diverse), ("collapsed", collapsed)]:
        p = torch.softmax(logits, -1)
        J5 = (p * rewards).sum(-1).mean()
        (g,) = torch.autograd.grad(J5, logits)
        H = float(-(p * torch.log(p.clamp(min=1e-45))).sum(-1).mean().detach())
        trap.append({"policy": name, "J": J5.item(), "entropy_nats": H, "grad_norm": float(g.norm())})
    rep.table(trap)
    rep.data["greedy_trap"] = trap
    rep.save()


if __name__ == "__main__":
    main()
