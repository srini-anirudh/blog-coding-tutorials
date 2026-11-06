"""Sections 11-12: counting bits. The capacity budget, and a compression bound that stops being vacuous.

Run:
    python labs/the-puzzle-of-overparameterization/scripts/10_bits_and_bounds.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from overparam.bits import capacity_budget, ladder
from overparam.common import Report, table

out = Report("10_bits_and_bounds")
b = capacity_budget()
out("Capacity budget, section 11 (3.6 bits/parameter from Morris et al. 2025; 2 bits/token is the post's assumption)")
out(f"7B params x 3.6 bits = {b['capacity_bits']:.3g} bits = {b['capacity_GB']:.2f} GB")
out(f"10T tokens x 2 bits  = {b['corpus_bits']:.3g} bits = {b['corpus_TB']:.2f} TB")
out(f"corpus / capacity = {b['corpus_over_capacity']:.0f}x; the two match at {b['tokens_where_they_match']:.3g} tokens")
grid = [{"params": p, "tokens": t, **capacity_budget(p, t)} for p in (1.25e8, 1e9, 7e9, 7e10) for t in (1e9, 1e11, 1e13)]
out(table(grid, {"params": ".3g", "tokens": ".3g", "corpus_over_capacity": ".3g"}))
out("A ratio below 1 is the memorization regime: the weights could hold the whole corpus.")
out()

out("Compression ladder, section 12: binary task y = 1[x1 x2 + 0.5 x3 > 0], x in R^10, n = 50,000, delta = 0.05")
out("bound = train error + sqrt((b ln 2 + ln(#rungs / delta)) / 2n). Test error is measured on 20,000 fresh points.")
for rl in (False, True):
    res = ladder(random_labels=rl)
    out(f"{'random' if rl else 'real'} labels, MLP 10-32-32-2 with {res['params']:,} parameters")
    out(table(res["rows"], {"rung": "", "bits": "", "train_err": ".4f", "test_err": ".4f", "bound": ".4f"}))
    best = min(res["rows"], key=lambda r: r["bound"])
    out(f"tightest: {best['rung']}, bound {best['bound']:.3f} vs measured test error {best['test_err']:.3f}")
    out()
    out.data["random_labels" if rl else "real_labels"] = res
out("On a two-class task a bound above 0.5 says nothing a coin flip does not.")
out.data["capacity"] = b
out.save()
