"""What the algebra says about binary rewards: closed-form advantages, wasted groups, difficulty weighting.

    python labs/how-reinforcement-learning-teaches-models-to-reason/scripts/06_binary_advantages.py
"""

import _path  # noqa: F401
import torch

from rlvrlab.advantages import (advantage_mass, binary_closed_form, dynamic_sampling_mask, group_advantages,
                                informative_prob)
from rlvrlab.common import Report

G = 8


def main() -> None:
    rep = Report("06_binary_advantages")
    rep.print(f"GRPO advantages in a group of {G} with k right. 'pop' divides by the population std (the post's")
    rep.print("algebra); 'unbiased' divides by the n-1 std, as TRL and C06 do.")
    rows = []
    for k in range(1, G):
        p = k / G
        a_r, a_w = binary_closed_form(p)
        r = torch.zeros(1, G)
        r[0, :k] = 1
        ub = group_advantages(r, "grpo", eps=0.0, unbiased_std=True)[0]
        rows.append([f"{k}/{G}", a_r, a_w, float(ub[0]), float(ub[-1]), abs(a_w / a_r) if p > 0.5 else a_r / abs(a_w)])
    rep.table(["p", "A_right (pop)", "A_wrong (pop)", "A_right (unbiased)", "A_wrong (unbiased)", "rare/common"], rows)
    rep.print()

    rep.print("Chance a group is not unanimous, so it carries any gradient: 1 - p^G - (1-p)^G")
    ps = [0.01, 0.05, 0.1, 0.3, 0.5, 0.9, 0.95, 0.99]
    gs = [4, 8, 16, 64]
    rep.table(["p"] + [f"G={g}" for g in gs], [[p] + [informative_prob(p, g) for g in gs] for p in ps])
    rep.print()

    rep.print(f"How hard one prompt pushes: sum |A_i| over the group, G={G}.")
    mrows = [[f"{k}/{G}", advantage_mass(k, G, "grpo"), advantage_mass(k, G, "mean")] for k in range(G + 1)]
    rep.table(["k right", "GRPO (z-score)", "Dr. GRPO (mean only)"], mrows)
    ratio_grpo = advantage_mass(1, G, "grpo") / advantage_mass(4, G, "grpo")
    ratio_mean = advantage_mass(1, G, "mean") / advantage_mass(4, G, "mean")
    rep.print(f"weight of a 1/8 prompt relative to a 4/8 prompt: GRPO {ratio_grpo:.3f}, mean-only {ratio_mean:.3f}")
    rep.print()

    gen = torch.Generator().manual_seed(0)
    n = 4096
    p = torch.distributions.Beta(0.4, 0.4).sample((n, 1))  # many prompts the model always or never solves
    r = (torch.rand(n, G, generator=gen) < p).float()
    keep = dynamic_sampling_mask(r)
    rep.print(f"Dynamic sampling, {n} prompts with solve rate ~ Beta(0.4, 0.4), G={G}:")
    rep.print(f"  groups with a gradient: {keep.float().mean():.3f}; all right: {(r.min(1).values == 1).float().mean():.3f};"
              f" all wrong: {(r.max(1).values == 0).float().mean():.3f}")
    rep.print(f"  rollouts spent per informative group without filtering: {G / keep.float().mean():.2f} (minimum {G})")
    rep.data = {"closed_form": rows, "advantage_mass": mrows, "informative_fraction": float(keep.float().mean()),
                "ratio_1of8_vs_4of8": {"grpo": ratio_grpo, "mean": ratio_mean}}
    rep.save()


if __name__ == "__main__":
    main()
