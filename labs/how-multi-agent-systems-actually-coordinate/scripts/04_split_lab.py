"""The post's split-decision lab (Figure 8), run in Python, plus one-factor sweeps.

    python labs/how-multi-agent-systems-actually-coordinate/scripts/04_split_lab.py
"""

import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mascoord.splitlab import PRESETS, evaluate, verdict  # noqa: E402
from mascoord.report import Report  # noqa: E402

out = Report("04_split_lab")

out("\n1. The eight preset scenarios")
out(f"{'scenario':<32} {'ctx':>6} {'ens':>6} {'hand':>6} {'quality':>8} {'price':>6} {'wall':>6} {'q/$':>6}  verdict")
for s in PRESETS:
    r = evaluate(s)
    out(f"{s.name:<32} {r['ctx_gain']:>6.3f} {r['ens_gain']:>6.3f} {r['handoff']:>6.3f} {r['quality']:>8.3f} "
        f"{r['cost']:>6.2f} {r['wall']:>6.2f} {r['per_dollar']:>6.2f}  {verdict(r, s)}")

out("\n2. Even N is penalised by the vote term: a tie counts as wrong (rho = 0, p = 0.62)")
base = PRESETS[3]
out("   " + "  ".join(f"N={n}: {evaluate(replace(base, N=n, rho=0.0))['ens_gain']:.3f}" for n in range(1, 9)))

out("\n3. Sweeps around the breadth-first research preset, one factor at a time")
for field, values in [("N", [1, 2, 3, 5, 8, 12, 16, 24]), ("press", [0.1, 0.3, 0.5, 0.7, 0.9]),
                      ("rho", [0.0, 0.25, 0.5, 0.75, 0.95]), ("delta", [0.0, 0.1, 0.2, 0.3, 0.45]),
                      ("tier", [0.05, 0.15, 0.25, 0.5, 1.0])]:
    out(f"   {field}:")
    for v in values:
        s = replace(base, **{field: v})
        r = evaluate(s)
        out(f"     {field}={v:<5} quality {r['quality']:.3f}  price {r['cost']:.2f}  wall {r['wall']:.2f}  "
            f"-> {verdict(r, s)}")
