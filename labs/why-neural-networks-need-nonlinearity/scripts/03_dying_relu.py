"""Section 03: a dead ReLU is an absorbing state. Leaky ReLU, ELU and GELU have no such state.

(a) Fig 03's experiment, trained for real. 60 hidden units learn two spirals with plain SGD. For the first 600
    steps every hidden bias is also pushed down by a fixed amount per step (the "negative bias drift"). Then the
    drift stops and training continues. A unit is "inactive" if its pre-activation is <= 0 on every training input.
    For ReLU that unit's gradient is exactly zero, so it can never come back. The other activations still pass a
    gradient on the negative side, so their inactive units can recover.
(b) The same absorbing state, with no drift at all: a large learning rate on a three-layer net.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))

from nonlin.data import two_spirals  # noqa: E402
from nonlin.mlp import DeepMLP  # noqa: E402
from nonlin.stats import Results  # noqa: E402

ACTS = ["relu", "leaky_relu", "leaky_relu_0.1", "elu", "gelu"]


@torch.no_grad()
def inactive_mask(model: DeepMLP, x: torch.Tensor) -> list[torch.Tensor]:
    """For each hidden layer, a bool per unit: is the pre-activation <= 0 for every input in x?"""
    masks, h = [], x
    for layer in model.hidden:
        z = layer(h)
        masks.append((z <= 0).all(dim=0))
        h = model.act(z)
    return masks


def drift_run(act: str, drift_steps: int, recover_steps: int, drift: float, lr: float, seed: int, x, y) -> dict:
    torch.manual_seed(seed)
    model = DeepMLP(2, 60, 2, 1, act, init="he")
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    layer = model.hidden[0]
    curve, frozen = [], None
    steps = drift_steps + recover_steps
    for step in range(steps + 1):
        if step % max(1, steps // 12) == 0:
            curve.append((step, inactive_mask(model, x)[0].float().mean().item()))
        if step == drift_steps:
            frozen = (inactive_mask(model, x)[0].clone(), layer.weight.detach().clone(), layer.bias.detach().clone())
        if step == steps:
            break
        opt.zero_grad()
        nn.functional.cross_entropy(model(x), y).backward()
        opt.step()
        if step < drift_steps:
            with torch.no_grad():
                layer.bias -= drift
    mid_mask, w_mid, b_mid = frozen
    end_mask = inactive_mask(model, x)[0]
    with torch.no_grad():
        acc = (model(x).argmax(1) == y).float().mean().item()
    # How far did the weights of the units that were inactive at the midpoint move in the second half?
    moved = 0.0
    if mid_mask.any():
        moved = max((layer.weight.detach()[mid_mask] - w_mid[mid_mask]).abs().max().item(),
                    (layer.bias.detach()[mid_mask] - b_mid[mid_mask]).abs().max().item())
    return {"curve": curve, "inactive_at_mid": int(mid_mask.sum()), "inactive_at_end": int(end_mask.sum()),
            "recovered": int((mid_mask & ~end_mask).sum()), "max_weight_change_of_mid_inactive": moved,
            "train_acc": acc}


def part_a(res: Results, drift_steps: int, recover_steps: int, drift: float, lr: float, seeds: list[int]) -> None:
    x, y = two_spirals(2000, seed=0)
    res.print(f"(a) 60 units, two spirals, SGD lr {lr}. Bias drift -{drift}/step for {drift_steps} steps, then "
              f"{recover_steps} steps without drift. Seeds {seeds}, counts summed over seeds.")
    res.print("max |dW| is how far the weights of the units inactive at the end of the drift moved afterwards.")
    res.print(f"{'activation':<11} {'inactive@mid':>13} {'inactive@end':>13} {'recovered':>10} "
              f"{'max |dW| of mid-inactive':>25} {'train acc':>10}")
    out = {}
    for act in ACTS:
        runs = [drift_run(act, drift_steps, recover_steps, drift, lr, s, x, y) for s in seeds]
        tot = {k: sum(r[k] for r in runs) for k in ("inactive_at_mid", "inactive_at_end", "recovered")}
        moved = max(r["max_weight_change_of_mid_inactive"] for r in runs)
        acc = sum(r["train_acc"] for r in runs) / len(runs)
        res.print(f"{act:<11} {tot['inactive_at_mid']:>13} {tot['inactive_at_end']:>13} {tot['recovered']:>10} "
                  f"{moved:>25.3g} {acc:>10.3f}")
        out[act] = {**tot, "units_total": 60 * len(seeds), "max_weight_change_of_mid_inactive": moved,
                    "train_acc": acc, "inactive_fraction_curve_seed0": runs[0]["curve"]}
    res.print("Inactive fraction over training, seed 0 (step: fraction):")
    for act in ACTS:
        res.print(f"  {act:<11} " + "  ".join(f"{s}:{f:.2f}" for s, f in out[act]["inactive_fraction_curve_seed0"]))
    res.data["drift"] = out


def part_b(res: Results, steps: int, lrs: list[float], seeds: list[int]) -> None:
    x, y = two_spirals(2000, seed=0)
    res.print(f"\n(b) No drift. 3 hidden layers of width 128, plain SGD, {steps} steps. Fraction of hidden units "
              f"inactive on every training input at the end, mean over seeds {seeds}")
    res.print(f"{'lr':>6} " + " ".join(f"{a:>11}" for a in ACTS))
    out = {}
    for lr in lrs:
        row = {}
        for act in ACTS:
            fr = []
            for s in seeds:
                torch.manual_seed(s)
                model = DeepMLP(2, 128, 2, 3, act, init="he")
                opt = torch.optim.SGD(model.parameters(), lr=lr)
                for _ in range(steps):
                    opt.zero_grad()
                    loss = nn.functional.cross_entropy(model(x), y)
                    if not torch.isfinite(loss):
                        break
                    loss.backward()
                    opt.step()
                fr.append(torch.cat(inactive_mask(model, x)).float().mean().item())
            row[act] = sum(fr) / len(fr)
        res.print(f"{lr:>6} " + " ".join(f"{row[a]:>11.3f}" for a in ACTS))
        out[str(lr)] = row
    res.data["large_lr"] = out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--drift-steps", type=int, default=600)
    p.add_argument("--recover-steps", type=int, default=1800)
    p.add_argument("--drift", type=float, default=0.002)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--out", type=Path, default=LAB / "results")
    args = p.parse_args()
    seeds = list(range(1 if args.quick else args.seeds))
    res = Results("03_dying_relu", args.out)
    if args.quick:
        part_a(res, 60, 60, 0.02, 0.1, seeds)
    else:
        part_a(res, args.drift_steps, args.recover_steps, args.drift, 0.1, seeds)
    part_b(res, 60 if args.quick else 600, [0.1, 1.0] if args.quick else [0.05, 0.2, 0.5, 1.0, 2.0], seeds[:3])
    res.save()


if __name__ == "__main__":
    main()
