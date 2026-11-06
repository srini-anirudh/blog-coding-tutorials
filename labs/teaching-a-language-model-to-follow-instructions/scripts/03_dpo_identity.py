"""Section "The DPO identity, derived slowly": every step of the derivation, checked with numbers.

Route 2 (fit a reward model, then take the KL-optimal policy) and route 3 (DPO, no reward model) are run on the
same comparisons. With one free logit per completion they land on the same policy.

    python labs/teaching-a-language-model-to-follow-instructions/scripts/03_dpo_identity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prefopt.bt import bt_prob, fit_bt  # noqa: E402
from prefopt.common import Report  # noqa: E402
from prefopt.tilt import fit_dpo_tabular, implicit_reward, log_partition, optimal_logp  # noqa: E402

P, K, BETA = 8, 6, 0.5


def tv(a_logp: torch.Tensor, b_logp: torch.Tensor) -> float:
    """Mean over prompts of the total-variation distance."""
    return (0.5 * (a_logp.exp() - b_logp.exp()).abs().sum(-1)).mean().item()


def sample_pairs(r: torch.Tensor, n: int, gen: torch.Generator):
    """n comparisons: a random prompt, two distinct completions, a Bradley-Terry label from the true reward."""
    x = torch.randint(0, P, (n,), generator=gen)
    i = torch.randint(0, K, (n,), generator=gen)
    j = (i + torch.randint(1, K, (n,), generator=gen)) % K
    i_wins = torch.rand(n, generator=gen, dtype=torch.float64) < bt_prob(r[x, i] - r[x, j])
    return x, torch.where(i_wins, i, j), torch.where(i_wins, j, i)


def main() -> None:
    rep = Report("03_dpo_identity")
    gen = torch.Generator().manual_seed(0)
    ref = torch.log_softmax(torch.randn(P, K, generator=gen, dtype=torch.float64), -1)
    r = torch.randn(P, K, generator=gen, dtype=torch.float64)
    rep.print(f"{P} prompts x {K} completions, beta = {BETA}, random reference and true reward r.")

    rep.print("\nSteps 2-3: r = beta log(pi*/ref) + beta log Z(x). Check it per prompt.")
    star = optimal_logp(ref, r, BETA)
    gap = implicit_reward(star, ref, BETA) - r  # should be -beta log Z(x), the same for every completion
    rows = [{"prompt": x, "-beta*logZ": -BETA * log_partition(ref[x], r[x], BETA).item(),
             "mean(r_hat - r)": gap[x].mean().item(), "spread(r_hat - r)": (gap[x].max() - gap[x].min()).item()}
            for x in range(4)]
    rep.table(rows, "{:.6f}")
    spread = (gap.max(-1).values - gap.min(-1).values).max().item()
    rep.print(f"largest within-prompt spread of r_hat - r over all {P} prompts: {spread:.2e}")
    rep.data["identity_spread"] = spread

    rep.print("\nSteps 4-5: the intractable term cancels inside Bradley-Terry.")
    i, j = torch.meshgrid(torch.arange(K), torch.arange(K), indexing="ij")
    r_hat = implicit_reward(star, ref, BETA)
    diff = (bt_prob(r_hat[:, i] - r_hat[:, j]) - bt_prob(r[:, i] - r[:, j])).abs().max().item()
    rep.print(f"max |sigma(r_hat_w - r_hat_l) - sigma(r_w - r_l)| over all {P * K * K} ordered pairs: {diff:.2e}")
    rep.data["bt_cancel"] = diff

    rep.print("\nStep 6 and the two routes, on the same sampled comparisons:")
    rep.print("  route 2: fit Bradley-Terry scores, then pi = ref * exp(score / beta) / Z")
    rep.print("  route 3: DPO directly on the policy logits")
    routes = []
    for n in [300, 3000, 30000, 300000]:
        x, w, l = sample_pairs(r, n, gen)
        scores = fit_bt(P * K, x * K + w, x * K + l, steps=3000).view(P, K)
        via_rm = optimal_logp(ref, scores, BETA)
        via_dpo = fit_dpo_tabular(ref, x, w, l, BETA, steps=3000)
        routes.append({"pairs": n, "TV(rm route, dpo route)": tv(via_rm, via_dpo),
                       "TV(dpo, pi*)": tv(via_dpo, star), "TV(ref, pi*)": tv(ref, star)})
    rep.table(routes, "{:.5f}")
    rep.print("The two routes agree to optimiser precision at every data size. Both approach pi* as data grows.")
    rep.data["routes"] = routes
    rep.save()


if __name__ == "__main__":
    main()
