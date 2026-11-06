"""Behavioural cloning, compounding error, DAgger, RL from outcomes, and static versus interactive evaluation.

Post sections: "Behavioural cloning and compounding error" (Figure 9), "Reinforcement learning from task outcomes"
and "Evaluating in interactive environments" (Figure 10, the step-budget nuance).

Demonstrations are clean: the expert never errs and no pop-up ever appears, as in recorded human data. At test
time a pop-up appears during each report with probability 0.5. The policy reads only pixels.
"""

import argparse
import copy
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from actlab import policy as P  # noqa: E402
from actlab.report import RUNS, Report, seed_threads  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--demos", type=int, default=200)
ap.add_argument("--bc-steps", type=int, default=800)
ap.add_argument("--dagger-iters", type=int, default=3)
ap.add_argument("--dagger-steps", type=int, default=400)
ap.add_argument("--rl-iters", type=int, default=60)
ap.add_argument("--eval", type=int, default=200)
args = ap.parse_args()
seed_threads()
torch.manual_seed(0)
rep = Report("06_bc_dagger_rl")
t0 = time.time()
Q = 0.5

rep("Part 1. The no-recovery model (1 - eps)^T, checked in the app with a noisy expert (1000 episodes each).")
rep("  T = 30 steps (6 reports). Budget x1.0 leaves no spare step; x2.0 lets the expert recover from slips.")
part1 = []
for eps in (0.02, 0.05):
    exact = P.noisy_expert_success(eps, budget_factor=1.0)
    slack = P.noisy_expert_success(eps, budget_factor=2.0)
    part1.append(dict(eps=eps, predicted=(1 - eps) ** 30, budget_1x=exact, budget_2x=slack))
    rep(f"  eps {eps:.2f}: (1-eps)^30 = {(1 - eps) ** 30:.3f}   measured, budget x1.0 = {exact:.3f}"
        f"   budget x2.0 = {slack:.3f}")

rep("")
rep("Part 2. Train the pixel policies.")
O, C, A = P.expert_demos(args.demos, seed=1)
Oh, Ch, Ah = P.expert_demos(50, seed=2)  # held-out demonstrations for static evaluation
rep(f"  {args.demos} clean demonstrations, {len(A)} (screenshot, action) pairs")
policies = {}
bc = P.fit(P.PixelPolicy(), O, C, A, steps=args.bc_steps, log=rep)
policies["BC"] = bc
rep(f"  BC done ({time.time() - t0:.0f} s)")
O4, C4, A4 = P.expert_demos(4 * args.demos, seed=3)
torch.manual_seed(0)
policies["BC, 4x demos"] = P.fit(P.PixelPolicy(), O4, C4, A4, steps=args.bc_steps)
rep(f"  BC with {4 * args.demos} demonstrations done ({time.time() - t0:.0f} s)")
policies["DAgger"] = P.dagger(copy.deepcopy(bc), O, C, A, iters=args.dagger_iters, popup_prob=Q,
                               steps=args.dagger_steps, log=rep)
rep(f"  DAgger done ({time.time() - t0:.0f} s)")
rl, hist = P.reinforce(copy.deepcopy(bc), iters=args.rl_iters, popup_prob=Q, kl_coef=1.0, log=rep)
policies["BC + RL"] = rl
rep(f"  BC + RL (KL 1.0 to the BC policy) done ({time.time() - t0:.0f} s)")
policies["BC + RL, no KL"], hist0 = P.reinforce(copy.deepcopy(bc), iters=args.rl_iters, popup_prob=Q, kl_coef=0.0,
                                                log=rep)
rep(f"  BC + RL without KL done ({time.time() - t0:.0f} s). RL used no expert labels, only the checker's 0/1 reward,")
rep("  with 10% of actions replaced by clicks on random on-screen elements for exploration.")
RUNS.mkdir(exist_ok=True)
torch.save({k: v.state_dict() for k, v in policies.items()}, RUNS / "policies.pt")

rep("")
rep(f"Part 3. Static versus interactive evaluation ({args.eval} live episodes per cell, greedy actions).")
rep("  static = next-action match on held-out expert histories; interactive = checker on the saved records")
hdr = f"  {'policy':16s} {'static':>7s} {'live, no pop-ups':>17s} {'live, pop-ups':>14s}"
rep(hdr)
table = {}
for name, pol in policies.items():
    s = P.static_accuracy(pol, Oh, Ch, Ah)
    clean = P.interactive_success(pol, n=args.eval, popup_prob=0.0)
    pop = P.interactive_success(pol, n=args.eval, popup_prob=Q)
    table[name] = dict(static=s, live_clean=clean, live_popups=pop)
    rep(f"  {name:16s} {s:7.3f} {clean:17.3f} {pop:14.3f}")

rep("")
rep("Part 4. Longer tasks: success against the number of reports (5 expert steps each), pop-ups on, budget x2.")
rep(f"  {'policy':16s} " + " ".join(f"{f'k={k} (T={5 * k})':>12s}" for k in (1, 2, 4, 6)))
horizon = {}
for name, pol in policies.items():
    horizon[name] = [P.interactive_success(pol, n=args.eval, n_reports=k, popup_prob=Q, seed=900 + k)
                     for k in (1, 2, 4, 6)]
    rep(f"  {name:16s} " + " ".join(f"{v:12.3f}" for v in horizon[name]))

rep("")
rep("Part 5. Step budget: 2 reports (10 expert steps), pop-ups on.")
rep(f"  {'policy':16s} " + " ".join(f"{f'x{b}':>8s}" for b in (1.0, 1.2, 1.5, 2.0, 3.0)))
budget = {}
for name, pol in policies.items():
    budget[name] = [P.interactive_success(pol, n=args.eval, n_reports=2, popup_prob=Q, budget_factor=b, seed=4242)
                    for b in (1.0, 1.2, 1.5, 2.0, 3.0)]
    rep(f"  {name:16s} " + " ".join(f"{v:8.3f}" for v in budget[name]))
rep(f"total time {time.time() - t0:.0f} s")
rep.save({"popup_prob": Q, "demos": args.demos, "noisy_expert": part1, "eval": table, "horizon": horizon,
          "budget": budget, "rl_sampled_success": hist, "rl_no_kl_sampled_success": hist0})
