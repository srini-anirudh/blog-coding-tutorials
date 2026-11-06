"""Section 13: why RL lives in post-training.

1. From a random initialisation the gradient is exactly zero. The task rewards one specific 8-token answer over 16
   actions, so a near-uniform policy succeeds with probability about 16^-8. Every rollout scores 0, every RLOO
   weight is 0, and every gradient is the zero vector. The weights do not move at all.
2. A short supervised stage on noisy demonstrations lifts the exact success probability to a few percent. The same
   RL loop then works.
3. The chance that a group of G samples contains at least one success, against 1 - (1 - p)^G.

    python labs/reinforcement-learning-from-first-principles/scripts/09_cold_start.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rlfp.estimators import baseline_weights, pg_loss  # noqa: E402
from rlfp.exact import pass_at_k  # noqa: E402
from rlfp.policy import TinyLM, prompt_batch, sample, token_logprobs  # noqa: E402
from rlfp.report import Report  # noqa: E402
from rlfp.tasks import exact_match_reward  # noqa: E402

A, T, PROMPT = 16, 8, 16
TARGET = torch.tensor([3, 14, 1, 5, 9, 2, 6, 5])


def p_target(policy: TinyLM) -> float:
    """Exact success probability: the target is the only correct answer, so p = pi(target)."""
    with torch.no_grad():
        return float(token_logprobs(policy, prompt_batch([PROMPT]), TARGET.unsqueeze(0)).sum().exp())


def rl(policy: TinyLM, steps: int, G: int, lr: float, gen: torch.Generator, every: int = 10):
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    rows, nonzero_grads, successes = [], 0, 0
    for step in range(steps + 1):
        if step % every == 0:
            rows.append({"step": step, "p_target": p_target(policy)})
        if step == steps:
            break
        pb = prompt_batch([PROMPT], G)
        acts = sample(policy, pb, T, gen)
        R = exact_match_reward(acts, TARGET)
        successes += int(R.sum())
        loss = pg_loss(token_logprobs(policy, pb, acts), baseline_weights(R, "group", G))
        opt.zero_grad()
        loss.backward()
        nonzero_grads += int(any(p.grad.abs().max() > 0 for p in policy.parameters()))
        opt.step()
    return rows, nonzero_grads, successes


def main() -> None:
    rep = Report("09_cold_start")
    torch.manual_seed(0)
    cold = TinyLM(A + 1, A, hidden=64)
    before = [p.detach().clone() for p in cold.parameters()]
    p0 = p_target(cold)
    rows, nz, succ = rl(cold, steps=200, G=64, lr=3e-3, gen=torch.Generator().manual_seed(0), every=100)
    moved = max(float((p - b).abs().max()) for p, b in zip(cold.parameters(), before))
    rep.print(f"1. Random init: exact p(target) = {p0:.2e} (16^-8 = {16.0 ** -8:.2e}). 200 RL steps x 64 rollouts: "
              f"{succ} successes, {nz} steps with a nonzero gradient, largest weight change {moved:.1e}")
    rep.data["cold"] = {"p0": p0, "successes": succ, "nonzero_grad_steps": nz, "max_weight_change": moved}

    # 2. Supervised warm start: imitate demonstrations in which each token is right with probability 0.6.
    torch.manual_seed(0)
    warm = TinyLM(A + 1, A, hidden=64)
    opt = torch.optim.Adam(warm.parameters(), lr=3e-3)
    gen = torch.Generator().manual_seed(1)
    sft_steps = 0
    while p_target(warm) < 0.02:
        demo = TARGET.repeat(64, 1)
        noise = torch.rand(demo.shape, generator=gen) > 0.6
        demo[noise] = torch.randint(0, A, (int(noise.sum()),), generator=gen)
        loss = -token_logprobs(warm, prompt_batch([PROMPT], 64), demo).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sft_steps += 1
    p_warm = p_target(warm)
    rep.print(f"\n2. {sft_steps} SFT steps on noisy demonstrations: exact p(target) = {p_warm:.4f}. Then the same RL loop:")
    rows, nz, succ = rl(warm, steps=100, G=64, lr=3e-3, gen=torch.Generator().manual_seed(2))
    rep.table(rows)
    rep.print(f"{nz} of 100 steps had a nonzero gradient; {succ} successes in 6400 rollouts")
    rep.data["warm"] = {"sft_steps": sft_steps, "p_after_sft": p_warm, "rl_curve": rows, "nonzero_grad_steps": nz}

    # 3. P(group contains a success) for policies of known p.
    rep.print("\n3. Share of groups with at least one success: measured on Bernoulli(p) rewards, 20,000 groups")
    gen = torch.Generator().manual_seed(3)
    rows = []
    for p in [0.0, 0.01, 0.05, 0.12]:
        for G in [4, 16, 64]:
            hits = (torch.rand(20_000, G, generator=gen) < p).any(1).float().mean()
            rows.append({"p": p, "G": G, "measured": float(hits), "1-(1-p)^G": float(pass_at_k(p, G))})
    rep.table(rows)
    rep.data["group_signal"] = rows
    rep.save()


if __name__ == "__main__":
    main()
