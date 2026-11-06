"""Elicit or create: RL sharpens the strategy distribution. pass@1 rises; pass@k at large k falls.

    python labs/how-reinforcement-learning-teaches-models-to-reason/scripts/08_pass_at_k.py
"""

import _path  # noqa: F401
import torch

from rlvrlab.common import Report
from rlvrlab.sharpening import SharpenConfig, base_logits, make_problems, pass_at_k_exact, pass_at_k_unbiased, train

KS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 1024]


def crossover(base: dict, rl: dict) -> int | None:
    for k in sorted(base):
        if base[k] >= rl[k]:
            return k
    return None


def main() -> None:
    rep = Report("08_pass_at_k")
    cfg = SharpenConfig()
    train_m, test_m = make_problems(cfg, seed=1), make_problems(cfg, seed=2)
    base = pass_at_k_exact(base_logits(cfg), test_m, KS)
    rep.print(f"{cfg.strategies} strategies, {cfg.problems} held-out test problems. Common strategy solves "
              f"{cfg.frac_common:.0%} of solvable problems; each problem has one rare strategy; "
              f"{cfg.frac_unsolvable:.0%} unsolvable.")
    rep.print(f"GRPO, {cfg.steps} steps x {cfg.batch} prompts x {cfg.group} samples, Adam lr {cfg.lr}. pass@k is exact.")
    rep.print()
    results, logs = {"base": base}, {}
    for beta in (0.0, 0.1, 1.0):
        theta, log = train(SharpenConfig(beta=beta), train_m, test_m, KS)
        results[f"rl beta={beta}"] = pass_at_k_exact(theta, test_m, KS)
        logs[beta] = log
    rows = [[name] + [r[k] for k in KS] for name, r in results.items()]
    rep.table(["policy"] + [f"pass@{k}" for k in KS], rows)
    for name, r in results.items():
        if name != "base":
            rep.print(f"{name}: base overtakes it at k = {crossover(base, r)}")
    rep.print()
    rep.print("The boundary narrows as training goes on (beta = 0):")
    rep.table(["step", "entropy", "pi(common)", "pass@1", "pass@16", "pass@256"],
              [[r["step"], r["entropy"], r["pi_common"], r["pass@1"], r["pass@16"], r["pass@256"]] for r in logs[0.0]])
    rep.print()

    # The instrument itself: the unbiased estimator from n samples against the exact value, base policy.
    gen = torch.Generator().manual_seed(0)
    s = test_m @ base_logits(cfg).softmax(-1)
    n = 256
    c = torch.binomial(torch.full_like(s, n), s, generator=gen)
    est = {k: sum(pass_at_k_unbiased(n, int(ci), k) for ci in c) / len(c) for k in (1, 16, 64, 256)}
    rep.print(f"Unbiased pass@k estimator from n={n} samples per problem vs exact (base policy):")
    rep.table(["k", "estimate", "exact"], [[k, est[k], base[k]] for k in est])
    rep.data = {"ks": KS, "results": results, "curve_beta0": logs[0.0], "estimator_check": est}
    rep.save()


if __name__ == "__main__":
    main()
