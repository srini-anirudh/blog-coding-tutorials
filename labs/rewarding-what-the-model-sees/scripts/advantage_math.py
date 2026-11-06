"""The post's GRPO arithmetic, computed: near-tie amplification, the whole-image box in a
group of misses, and how often a group carries no learning signal.

    python labs/rewarding-what-the-model-sees/scripts/advantage_math.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from visrl.grpo import (difficulty_filter, group_advantages, p_zero_advantage, sample_pass_rates,  # noqa: E402
                        useful_fraction, zero_advantage)
from visrl.report import table, write  # noqa: E402


def fmt(x: torch.Tensor) -> str:
    return f"{float(x[-1]):+.3f} (odd one out), {float(x[0]):+.3f} (the other seven)"


def groups() -> tuple[str, dict]:
    near_tie = torch.tensor([[0.30] * 7 + [0.31]])
    box = torch.tensor([[0.0] * 7 + [0.2]])
    rows = [
        ["7 x IoU 0.30, 1 x IoU 0.31", "std (population)", fmt(group_advantages(near_tie)[0])],
        ["7 x IoU 0.30, 1 x IoU 0.31", "std (sample, K-1)",
         fmt(((near_tie - near_tie.mean()) / near_tie.std())[0])],
        ["7 x IoU 0.30, 1 x IoU 0.31", "no std (Dr. GRPO)", fmt(group_advantages(near_tie, scale_by_std=False)[0])],
        ["7 x IoU 0.30, 1 x IoU 0.31", "rounded to 0.1 first",
         fmt(group_advantages(torch.round(near_tie * 10) / 10)[0])],
        ["7 misses (0), 1 whole-image box (0.2)", "std (population)", fmt(group_advantages(box)[0])],
    ]
    text = (f"Group mean of the box group: {float(box.mean()):.3f}\n\n"
            + table(rows, ["group of K = 8 rewards", "normalisation", "advantages"]))
    return text, {"near_tie_adv": group_advantages(near_tie)[0].tolist(), "box_adv": group_advantages(box)[0].tolist()}


def zero_table(g: torch.Generator) -> tuple[str, dict]:
    ps, ks = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95], [4, 8, 16]
    rows, data = [], {}
    for p in ps:
        row = [p]
        for k in ks:
            rewards = (torch.rand(100_000, k, generator=g) < p).float()
            mc = float(zero_advantage(rewards).float().mean())
            row.append(f"{p_zero_advantage(p, k):.4f} / {mc:.4f}")
            data[f"p={p},K={k}"] = {"closed_form": p_zero_advantage(p, k), "monte_carlo": mc}
        rows.append(row)
    text = ("P(zero advantage) = p^K + (1 - p)^K, closed form / Monte Carlo over 100,000 groups\n\n"
            + table(rows, ["pass rate p", *[f"K = {k}" for k in ks]]))
    return text, data


def batch_table(g: torch.Generator) -> tuple[str, dict]:
    """A batch of 200 prompts, pass rates spread around a mean; filter on pass rates ESTIMATED from 16 rollouts."""
    rows, data = [], {}
    for mean in [0.3, 0.5, 0.7, 0.9]:
        torch.manual_seed(int(mean * 100))
        p = sample_pass_rates(200, mean)
        est = (torch.rand(200, 16, generator=g) < p[:, None]).float().mean(1)       # Vision-G1: 16 rollouts
        kept = p[(est >= 0.2) & (est <= 0.8)]
        oracle_kept = difficulty_filter(p)
        u_all, u_kept = useful_fraction(p, 8, g), useful_fraction(kept, 8, g)
        rows.append([mean, f"{u_all:.2f}", len(kept), f"{u_kept:.2f}", len(oracle_kept)])
        data[str(mean)] = {"useful_unfiltered": u_all, "kept": len(kept), "useful_filtered": u_kept}
    text = ("200 prompts, pass rates ~ Beta around the mean (concentration 4), K = 8 binary rewards\n\n"
            + table(rows, ["mean pass rate", "useful share, no filter", "prompts kept by 0.2-0.8 filter",
                           "useful share after filter", "kept if true p were known"]))
    return text, data


def main() -> None:
    g = torch.Generator().manual_seed(0)
    parts = [groups(), zero_table(g), batch_table(g)]
    write("advantage_math", "\n\n".join(t for t, _ in parts),
          {"groups": parts[0][1], "zero_advantage": parts[1][1], "batch": parts[2][1]})


if __name__ == "__main__":
    main()
