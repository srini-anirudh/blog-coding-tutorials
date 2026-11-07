"""Sections "Online versus offline" and Nuance 02 (ii): the DPO loss is silent off the data's support.

Same pool, reference and gold reward as 04_overoptimization.py. The policy is log-linear on the pool:
pi(y) is proportional to ref(y) exp(theta . features(y)), so responses share parameters as in a real model.
Both runs minimise the same DPO loss against the same frozen reference, with the same label budget.
  offline: every pair is drawn from ref and labelled once, then DPO runs to convergence.
  online:  the budget is split over rounds; each round draws fresh pairs from the current policy, labels them with
           the gold Bradley-Terry annotator, adds them to the dataset and continues training.
The only difference is where the completions come from. This is a toy, not the Xu, Tajwar or Tang experiments.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from prefopt.dpo import SoftmaxPolicy, train_dpo  # noqa: E402
from prefopt.goodhart import make_world  # noqa: E402
from prefopt.report import Report  # noqa: E402
from prefopt.tilt import kl, tilt  # noqa: E402

BETA = 0.1


def label(world, a, b, g):
    a_wins = torch.rand(a.numel(), generator=g, dtype=torch.float64) < torch.sigmoid(world.gold[a] - world.gold[b])
    return torch.where(a_wins, a, b), torch.where(a_wins, b, a)


def draw(p, m, g):
    return torch.multinomial(p, m, replacement=True, generator=g), torch.multinomial(p, m, replacement=True, generator=g)


def evaluate(world, pol, seen):
    p = pol.log_probs().exp().detach()
    return {"gold_gain": ((p - world.ref) * world.gold).sum().item(), "kl": kl(p, world.ref).item(),
            "length_gain": ((p - world.ref) * world.feats[:, 0]).sum().item(),
            "mass_unseen": p[~seen].sum().item(), "ref_mass_unseen": world.ref[~seen].sum().item()}


def run(world, budget: int, rounds: int, seed: int, steps_per_round: int = 300):
    g = torch.Generator().manual_seed(seed)
    pol = SoftmaxPolicy(world.ref.log(), world.feats)
    ref = pol.log_probs().detach()  # frozen for every round
    seen = torch.zeros(world.ref.numel(), dtype=torch.bool)
    w_all, l_all = torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)
    per_round = budget // rounds
    for _ in range(rounds):
        sampler = pol.log_probs().exp().detach()  # rounds = 1: this is ref, so the run is offline
        w, l = label(world, *draw(sampler, per_round, g), g)
        w_all, l_all = torch.cat([w_all, w]), torch.cat([l_all, l])
        seen[w] = seen[l] = True
        # A fresh optimiser each round, against the same frozen reference (the initial policy).
        train_dpo(pol, w_all, l_all, BETA, steps_per_round, lr=0.05, ref=ref)
    return evaluate(world, pol, seen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--name", default="07_online_vs_offline")
    args = ap.parse_args()
    rep = Report(args.name)
    world = make_world()
    star = tilt(world.ref, world.gold, BETA)
    rep.line(f"beta = {BETA}. Exact KL-optimum for the gold reward: gold gain "
             f"{((star - world.ref) * world.gold).sum().item():.3f}, KL {kl(star, world.ref).item():.3f}")
    rep.line("Offline uses 1 round. Online uses the same label budget split over 10 rounds. Means over seeds.\n")
    rows, data = [], []
    for budget in (500, 2000, 8000):
        for name, rounds in (("offline", 1), ("online", 10)):
            res = [run(world, budget, rounds, s, steps_per_round=3000 // rounds) for s in range(args.seeds)]
            mean = {k: sum(r[k] for r in res) / len(res) for k in res[0]}
            data.append({"budget": budget, "mode": name, **mean, "per_seed": res})
            rows.append([budget, name, mean["gold_gain"], mean["kl"], mean["length_gain"], mean["ref_mass_unseen"],
                         mean["mass_unseen"]])
    rep.table(["labels", "mode", "gold gain", "KL(pi||ref)", "length gain", "ref mass on unlabelled",
               "policy mass on unlabelled"], rows)
    rep.line("\nBoth modes train 3000 full-batch Adam steps in total. 'Unlabelled' means the completion never")
    rep.line("appeared in any labelled pair. Gold's length term peaks at +1 over the ref mean.")
    rep.data = {"beta": BETA, "runs": data}
    rep.save()


if __name__ == "__main__":
    main()
