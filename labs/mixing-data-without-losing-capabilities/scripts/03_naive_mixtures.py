"""Proportional, equal, temperature and capped mixing on 24 sources: one knob, two failure modes (post section 5).

Pure arithmetic, no training. Sizes run from 700 to 2,000,000 examples, evenly spaced in log space. For each rule
it prints the effective number of sources exp(H(p)), the largest source's share, and how many epochs the
smallest source gets under a one-million-example budget.

    python labs/mixing-data-without-losing-capabilities/scripts/03_naive_mixtures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mixlab import mixtures as mx  # noqa: E402
from mixlab.runs import Report  # noqa: E402

BUDGET = 1_000_000
MEMORISE = 4.0  # epochs above which we flag a source as "about to be memorised"


def row(name, p, sizes):
    ep = mx.epochs(p, sizes, BUDGET)
    return [
        name,
        f"{mx.n_eff(p):.1f}",
        f"{100 * max(p):.1f}%",
        f"{100 * sum(sorted(p)[-3:]):.1f}%",
        f"{ep[0]:.2f}",
        f"{max(ep):.2f}",
        sum(e > MEMORISE for e in ep),
    ]


def main() -> None:
    rep = Report("03_naive_mixtures")
    sizes = mx.log_spaced_sizes()
    rep(f"24 sources, sizes {sizes[0]:,} to {sizes[-1]:,} (log-spaced), total {sum(sizes):,}; budget {BUDGET:,} examples")
    rep("")
    rules = [("proportional (alpha=1)", mx.proportional(sizes))]
    for a in (0.7, 0.5, 0.3):
        rules.append((f"temperature alpha={a}", mx.temperature(sizes, a)))
    rules.append(("equal (alpha=0)", mx.equal(sizes)))
    for cap in (10_000, 50_000):
        rules.append((f"capped kappa={cap:,}", mx.capped(sizes, cap)))
    rep.table(
        ["rule", "N_eff", "largest", "top 3", "epochs, smallest", "max epochs", f"sources > {MEMORISE:.0f} epochs"],
        [row(n, p, sizes) for n, p in rules],
    )
    rep("")
    small = mx.epochs(mx.equal([700] * 24), [700] * 24, BUDGET)[0]
    rep(f"Post's example: a 700-example task in a 24-task equal mixture, 1M budget -> {small:.1f} epochs")
    prop_ep = mx.epochs(mx.proportional(sizes), sizes, BUDGET)
    rep(f"Proportional sees every source the same number of times: {min(prop_ep):.4f} to {max(prop_ep):.4f} epochs")
    rep("")
    rep("The counting question: split one 'code' source into 4 sub-sources and mix equally.")
    before = mx.equal([1] * 6)
    after = mx.equal([1] * 9)
    rep(f"  code share with 6 tasks: {100 * before[0]:.1f}%  ->  with code split in 4 (9 tasks): {400 * after[0]:.1f}%")
    rep.data = {"sizes": sizes, "rules": {n: {"p": p, "epochs": mx.epochs(p, sizes, BUDGET)} for n, p in rules}}
    rep.save()


if __name__ == "__main__":
    main()
