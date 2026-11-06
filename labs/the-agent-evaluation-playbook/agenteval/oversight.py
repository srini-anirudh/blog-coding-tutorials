"""Reversibility thresholds and a reviewer with finite capacity."""

from __future__ import annotations

import numpy as np


def p_star(U: float, L: float, c_r: float | None = None) -> float:
    """Confidence above which acting alone beats not acting: L_eff / (U + L_eff), with L_eff = min(c_r, L)."""
    l_eff = L if c_r is None else min(c_r, L)
    return l_eff / (U + l_eff)


# U is the gain from a correct action. L is the loss from a wrong one. c_r is the cost of undoing it.
TIERS = {
    "read-only": dict(U=1.0, L=0.0),
    "reversible, cheap undo": dict(U=1.0, L=100.0, c_r=0.25),
    "reversible, costly undo": dict(U=1.0, L=100.0, c_r=5.0),
    "irreversible, L/U=100": dict(U=1.0, L=100.0),
}


def threshold_policies(n: int, rng, review_cost: float = 0.01, conf_ab=(19.0, 1.0)) -> list[dict]:
    """Compare three acting policies on calibrated actions in each tier.

    The agent's confidence p ~ Beta(conf_ab) and the action is correct with probability p (perfect calibration,
    which is generous). Not acting alone is worth 0, the post's comparison point, minus `review_cost` for the
    person who takes the escalation. The best threshold is then p* shifted down slightly by the review cost.
    """
    rows = []
    for tier, kw in TIERS.items():
        conf = rng.beta(*conf_ab, n)
        correct = rng.random(n) < conf
        l_eff = kw["L"] if "c_r" not in kw else min(kw["c_r"], kw["L"])
        thr = p_star(**kw)
        for name, act in (("always act", np.ones(n, bool)), ("act if p > 0.9", conf > 0.9),
                          ("act if p > p*", conf > thr)):
            value = np.where(act, np.where(correct, kw["U"], -l_eff), -review_cost)
            below = conf < thr
            rows.append(dict(tier=tier, p_star=thr, policy=name, utility=float(value.mean()),
                             escalated=float(1 - act.mean()),
                             below_threshold_acts=float((act & below).sum() / max(act.sum(), 1)),
                             escalation_recall=float((~act & below).sum() / max(below.sum(), 1))))
    return rows


def escalation_sweep(e_grid, n: int, rng, q0=0.9, C=0.2, kappa=0.02, ranked=True, conf_ab=(9.0, 1.0)) -> list[dict]:
    """Escalate a fraction e of actions to a reviewer whose per-item catch rate is q(e) = q0 / (1 + e/C).

    With `ranked=False` the escalated actions are a random subset (the post's model, where the fraction of errors
    caught is e * q(e)). With `ranked=True` the agent escalates its least confident actions first. Harm per action is
    the rate of errors that reach the world plus `kappa` per escalation.
    """
    conf = rng.beta(*conf_ab, n)
    wrong = rng.random(n) > conf
    order = np.argsort(conf) if ranked else rng.permutation(n)
    out = []
    for e in e_grid:
        m = int(round(e * n))
        esc = order[:m]
        q = q0 / (1 + e / C)
        caught = int((wrong[esc] & (rng.random(m) < q)).sum())
        tp = int(wrong[esc].sum())
        prec = tp / m if m else 0.0
        rec = tp / wrong.sum()
        out.append(dict(e=float(e), q=q, caught_frac=caught / wrong.sum(), harm=(wrong.sum() - caught) / n + kappa * e,
                        precision=prec, recall=rec, f1=2 * prec * rec / (prec + rec) if tp else 0.0))
    return out


def harm_formula(e, eps, q0=0.9, C=0.2, kappa=0.02):
    """The post's H(e) = eps (1 - e q(e)) + kappa e, for random escalation."""
    e = np.asarray(e, float)
    return eps * (1 - e * q0 / (1 + e / C)) + kappa * e
