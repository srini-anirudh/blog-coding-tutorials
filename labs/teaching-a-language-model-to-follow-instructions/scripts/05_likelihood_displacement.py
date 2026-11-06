"""Section "The failure mode hiding in plain sight": DPO lowers the chosen response's probability while its loss
falls, and the mass goes to completions that never appear in the data. Then two patches that stop it.

A toy policy over 1016 completions with shared 64-dim features: 8 (chosen, rejected) pairs plus 1000 others.
The similarity knob is the cosine between the chosen and rejected features, the post's Figure 8 slider.

    python labs/teaching-a-language-model-to-follow-instructions/scripts/05_likelihood_displacement.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prefopt.bandit import similar_pairs_world, train_dpo  # noqa: E402
from prefopt.common import Report  # noqa: E402

PAIRS, OTHERS, DIM, BETA, STEPS, LR = 8, 1000, 64, 0.1, 400, 0.05


def summary(h: list[dict]) -> dict:
    last = h[-1]
    return {"loss": last["loss"], "dlogp_chosen": last["d_logp_chosen"], "dlogp_rejected": last["d_logp_rejected"],
            "mass_chosen": last["mass_chosen"], "mass_elsewhere": last["mass_elsewhere"], "KL": last["kl"]}


def main() -> None:
    rep = Report("05_likelihood_displacement")
    rep.print(f"{PAIRS} pairs, {OTHERS} unrelated completions, features in R^{DIM}, uniform reference, "
              f"beta = {BETA}, {STEPS} full-batch Adam steps at lr {LR}.")
    rep.print(f"At the start the chosen set holds {PAIRS / (2 * PAIRS + OTHERS):.4f} of the mass, "
              f"the unrelated completions {OTHERS / (2 * PAIRS + OTHERS):.4f}.")
    rep.print("dlogp = mean change in log-prob versus the reference, over the 8 chosen (or rejected) responses.")

    rep.print("\n1. Plain DPO, trajectory, for two similarities.")
    for sim in (0.0, 0.9):
        world, w, l = similar_pairs_world(PAIRS, OTHERS, DIM, sim)
        _, h = train_dpo(world, w, l, beta=BETA, steps=STEPS, lr=LR, log_every=50)
        rep.print(f"\nsimilarity {sim}")
        rep.table([{k: x[k] for k in ("step", "loss", "d_logp_chosen", "d_logp_rejected", "mass_chosen",
                                      "mass_elsewhere")} for x in h], "{:.3f}")
        rep.data[f"trajectory_sim{sim}"] = h

    rep.print("\n2. End of training, plain DPO and two patches, across similarity.")
    rep.print("   DPOP: DPO-Positive, lambda = 5.  NLL: an extra -0.1 * log pi(y_w) term (RPO-style).")
    rows = []
    for sim in (0.0, 0.5, 0.9, 0.99):
        world, w, l = similar_pairs_world(PAIRS, OTHERS, DIM, sim)
        for name, kw in (("DPO", {}), ("DPOP", {"dpop_lambda": 5.0}), ("DPO+NLL", {"nll_alpha": 0.1})):
            _, h = train_dpo(world, w, l, beta=BETA, steps=STEPS, lr=LR, log_every=STEPS, **kw)
            rows.append({"similarity": sim, "method": name, **summary(h)})
    rep.table(rows, "{:.3f}")
    rep.data["end"] = rows
    rep.print("\nPlain DPO: the loss falls and the chosen log-prob falls with it, more steeply as similarity grows.")
    rep.print("Nearly all mass ends on completions outside the data. DPOP keeps the chosen at or just above the")
    rep.print("reference. The NLL term raises the chosen by over 4 nats; at similarity 0.99 it raises the rejected")
    rep.print("too, because the two share almost all their features.")
    rep.save()


if __name__ == "__main__":
    main()
