"""Section 9: why you cannot take a big step.

1. A small step in theta can be a large step in pi. 200 random parameter directions, all of the same length, give
   very different exact KL(pi_new || pi_old).
2. Entropy collapse. The same RLOO policy gradient, with no KL and no clip, at four learning rates. Everything is
   measured exactly over all completions: pass@1, pass@16, the entropy of the completion distribution, and the KL
   moved per step. Targets 4 and 20 are the rare prompts (about 1% of completions are correct at the start).

    python labs/reinforcement-learning-from-first-principles/scripts/05_trust_region.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rlfp.exact import kl, pass_at_k  # noqa: E402
from rlfp.report import Report  # noqa: E402
from rlfp.train import TrainConfig, make_policy, sum_task, train  # noqa: E402


def kl_per_direction(rep: Report, radius: float = 0.5, n_dirs: int = 200) -> None:
    task = sum_task()
    untrained = make_policy(task, hidden=64, seed=0)
    trained = make_policy(task, hidden=64, seed=0)
    train(TrainConfig(method="rloo", steps=100, lr=3e-3, eval_every=100), policy=trained)
    rep.print(f"1. {n_dirs} random parameter directions, ||delta theta|| = {radius}, exact KL(new || old) in nats")
    rows = [_directions(task, untrained, radius, n_dirs, "untrained"),
            _directions(task, trained, radius, n_dirs, "after 100 RLOO steps")]
    rep.table(rows, fmt="{:.3g}")
    out = {"rows": rows}
    for p0 in (0.5, 0.99):
        logit = torch.log(torch.tensor(p0 / (1 - p0)))
        new = torch.sigmoid(logit + 1)
        k2 = float(new * torch.log(new / p0) + (1 - new) * torch.log((1 - new) / (1 - p0)))
        rep.print(f"   one logit +1.0 on a two-way choice at p = {p0}: p -> {float(new):.4f}, KL = {k2:.5f}")
        out[f"logit_step_kl_p{p0}"] = k2
    rep.data["kl_per_direction"] = out


@torch.no_grad()
def _directions(task, policy, radius, n_dirs, label) -> dict:
    params = list(policy.parameters())
    base = [lp.clone() for lp in task.logps(policy)]
    gen = torch.Generator().manual_seed(0)
    kls = []
    for _ in range(n_dirs):
        d = [torch.randn(p.shape, generator=gen) for p in params]
        norm = torch.sqrt(sum((x ** 2).sum() for x in d))
        for p, x in zip(params, d):
            p.add_(radius * x / norm)
        kls.append(float(torch.stack([kl(a, b) for a, b in zip(task.logps(policy), base)]).mean()))
        for p, x in zip(params, d):
            p.sub_(radius * x / norm)
    k = torch.tensor(kls)
    q = {"policy": label, **{name: float(k.quantile(v)) for name, v in [("min", 0.0), ("median", 0.5), ("max", 1.0)]}}
    q["max/min"] = q["max"] / q["min"]
    return q


def collapse(rep: Report) -> None:
    rep.print("\n2. RLOO, G = 8, 300 steps, Adam, no KL penalty, no clip. Exact metrics at the end.")
    rows, curves = [], {}
    for lr in [1e-3, 3e-3, 1e-2, 3e-2]:
        out = train(TrainConfig(method="rloo", steps=300, lr=lr, eval_every=25, track_step_kl=True))
        hist = [h for h in out["history"] if "pass@1" in h]
        step_kl = [h["step_kl"] for h in out["history"] if "step_kl" in h]
        last, first = hist[-1], hist[0]
        rare_start = [float(pass_at_k(first["per_prompt_pass@1"][i], 16)) for i in (0, 4)]
        rare_end = [float(pass_at_k(last["per_prompt_pass@1"][i], 16)) for i in (0, 4)]
        rows.append({"lr": lr, "pass@1": last["pass@1"], "pass@16": last["pass@16"],
                     "seq_entropy": last["seq_entropy"], "max_step_kl": max(step_kl),
                     "rare_pass@16_start": sum(rare_start) / 2, "rare_pass@16_end": sum(rare_end) / 2})
        curves[str(lr)] = [{k: h[k] for k in ("step", "pass@1", "pass@16", "seq_entropy", "per_prompt_pass@1")}
                           for h in hist]
    rep.print(f"at step 0: pass@1 {hist[0]['pass@1']:.4f}, pass@16 {hist[0]['pass@16']:.4f}, "
              f"seq_entropy {hist[0]['seq_entropy']:.3f} nats (uniform would be {5 * torch.log(torch.tensor(6.)):.3f})")
    rep.table(rows)
    rep.print("rare = targets 4 and 20, averaged. pass@16 = 1 - (1 - p)^16 from each prompt's exact pass rate.")
    rep.data["collapse"] = {"rows": rows, "curves": curves}


def main() -> None:
    rep = Report("05_trust_region")
    kl_per_direction(rep)
    collapse(rep)
    rep.save()


if __name__ == "__main__":
    main()
