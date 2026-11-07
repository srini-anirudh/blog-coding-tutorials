"""Section 3: SFT is behaviour cloning. Compounding error, and a DAgger-style fix, on a toy.

A. The classical construction: with per-step error eps on expert states and no recovery data, the cost is
   about eps*T^2/2. By construction this is exact, so it checks the arithmetic of the argument, not nature.
B. A learned policy (an MLP) cloned from expert states in a thin band around a ring: cost against horizon for
   behaviour cloning, noise-injected cloning (DART-style recovery data) and DAgger iterations.
Cost is bounded: 1 per step spent more than 0.1 from the ring.
"""

import argparse

import _common
import numpy as np
import torch
from sftlab import bc


def part_a(out):
    out("A. Worst-case chain. Cost of the cloned policy for per-step error eps, 20,000 simulated rollouts each")
    out(f"   {'eps':>7} {'T':>5} {'simulated':>10} {'eps*T^2/2':>10} {'exact sum':>10} {'eps*T (recovery)':>17} {'simulated recovery':>19}")
    for eps in [0.001, 0.005]:
        for T in [10, 20, 50, 100, 200]:
            sim = bc.tabular_worst_case(eps, T)
            exact = sum(1 - (1 - eps) ** t for t in range(1, T + 1))
            rec = bc.tabular_worst_case(eps, T, recover=True)
            out(f"   {eps:7.3f} {T:5d} {sim:10.3f} {eps * T * T / 2:10.3f} {exact:10.3f} {eps * T:17.3f} {rec:19.3f}")
    Ts = [10, 20, 50, 100, 200]
    s = bc.loglog_slope(Ts, [bc.tabular_worst_case(0.001, T) for T in Ts])
    s2 = bc.loglog_slope(Ts, [bc.tabular_worst_case(0.001, T, recover=True) for T in Ts])
    out(f"   log-log slope of cost against T at eps = 0.001: no recovery {s:.2f}, with recovery {s2:.2f}")
    out("   The T^2 is built into the construction (an error is never undone). The point of the table is the shape the")
    out("   post's bound describes, and that `eps*T^2/2` is the small-eps*T limit of the exact sum.")
    out("")


def part_b(out, seeds, Ts, n_expert, dagger_iters, noise):
    out(f"B. Learned policy on the ring task. Process noise {noise}, expert data {n_expert} trajectories of 100 steps,")
    out("   MLP 2-64-64-2 fitted by least squares. Cost = steps spent more than 0.1 from the ring, mean of 200 rollouts.")
    gen0 = torch.Generator().manual_seed(0)
    S, _ = bc.rollout(bc.expert_action, 300, 100, noise, gen0)
    dev = float((S.norm(dim=-1) - 1).abs().max())
    out(f"   Expert states stay within {dev:.3f} of the ring (the band the data covers).")
    res = {"expert": [], "bc": [], "bc_noise": [], **{f"dagger_{k}": [] for k in range(1, dagger_iters + 1)}}
    for seed in seeds:
        g = torch.Generator().manual_seed(seed)
        S, _ = bc.rollout(bc.expert_action, n_expert, 100, noise, g)
        net = bc.fit(S, seed=seed)
        res["bc"].append([bc.mean_cost(net, T, noise, seed) for T in Ts])
        res["expert"].append([bc.mean_cost(bc.expert_action, T, noise, seed) for T in Ts])
        Sn, _ = bc.rollout(bc.expert_action, n_expert, 100, noise, g, action_noise=0.05)  # noisy execution, clean labels
        res["bc_noise"].append([bc.mean_cost(bc.fit(Sn, seed=seed), T, noise, seed) for T in Ts])
        agg = S.reshape(-1, 2)
        cur = net
        for k in range(1, dagger_iters + 1):
            St, _ = bc.rollout(cur, n_expert, 100, noise, torch.Generator().manual_seed(1000 * seed + k))
            agg = torch.cat([agg, St.reshape(-1, 2)])
            cur = bc.fit(agg, seed=seed * 10 + k)
            res[f"dagger_{k}"].append([bc.mean_cost(cur, T, noise, seed) for T in Ts])
    out(f"   {len(seeds)} seeds (each seed re-draws the expert data and re-fits). Mean cost, with min to max over seeds in brackets.")
    out(f"   {'policy':<26}" + "".join(f"{'T=' + str(T):>22}" for T in Ts))
    names = {"expert": "expert", "bc": "behaviour cloning", "bc_noise": "cloning, noisy execution"}
    names.update({f"dagger_{k}": f"DAgger, {k} iteration{'s' if k > 1 else ''}" for k in range(1, dagger_iters + 1)})
    for k, label in names.items():
        a = np.array(res[k])
        out(f"   {label:<26}" + "".join(f"{a[:, i].mean():9.2f} [{a[:, i].min():5.1f},{a[:, i].max():6.1f}]" for i in range(len(Ts))))
    bcm = np.array(res["bc"]).mean(0)
    out("")
    out("   Local log-log slope of behaviour-cloning cost between consecutive horizons (2 would be the T^2 shape; 1 linear):")
    for i in range(len(Ts) - 1):
        s = np.log(bcm[i + 1] / max(bcm[i], 1e-9)) / np.log(Ts[i + 1] / Ts[i])
        out(f"     T = {Ts[i]} to {Ts[i+1]}: {s:.2f}   (mean cost {bcm[i]:.2f} to {bcm[i+1]:.2f})")
    out("   Reading. Behaviour cloning is a coin flip per seed: one seed's policy never leaves the ring, another leaves")
    out("   within 25 steps. Averaged over seeds the cost grows faster than linearly in T for most of the range (local")
    out("   slopes 1.0 to 1.9, noisy with 5 seeds), and it cannot keep the slope of 2 because cost is capped at T. The")
    out("   data never showed the way back to the ring, so a rollout that leaves stays out. Recovery data fixes it here:")
    out("   executing noisy actions while labelling with the clean expert action (the DART idea) gave zero cost at every")
    out("   horizon, and DAgger (label the states the learner visits, refit on the aggregate) took the mean cost to at most")
    out("   1.4 steps of 400 by iteration 2 to 4. It is not monotone in the iteration (1 iteration was still bad, 3 was")
    out("   worse than 2) because every iteration re-fits from scratch. The slope of the cost is not a clean test of the")
    out("   bound: the O(eps T^2) result is worst case, and this is one small task.")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--n-expert", type=int, default=50)
    ap.add_argument("--dagger-iters", type=int, default=4)
    ap.add_argument("--noise", type=float, default=0.002)
    a = ap.parse_args()
    out = _common.Report("03_behaviour_cloning", "cpu")
    part_a(out)
    res = part_b(out, a.seeds, [25, 50, 100, 200, 400], a.n_expert, a.dagger_iters, a.noise)
    out.save()
    _common.save_json("03_behaviour_cloning", {"cost_by_policy": res, "horizons": [25, 50, 100, 200, 400]}, "cpu")


if __name__ == "__main__":
    main()
