"""Section 12: scoring a smooth quantity all-or-nothing amplifies it (Figure 13), measured on real models.

Every sweep run was scored on the same 1M validation tokens (teacher forced) for:
  exact1  : next-token accuracy (the smooth, per-token quantity)
  exact k : the next k tokens are ALL predicted right, an exact-match score on a k-token answer.
No new training: this reads results/sweep.jsonl.

    python labs/scaling-laws-from-first-principles/scripts/08_emergence.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))


def main() -> None:
    lines = (LAB / "results" / "sweep.jsonl").read_text().splitlines()
    runs = [json.loads(x) for x in lines if x.startswith("{")]
    print(lines[0].replace("# hardware:", "# sweep hardware:"))
    print("# command: python scripts/08_emergence.py   (reads the sweep, CPU)\n")
    runs.sort(key=lambda r: -r["val_loss"])
    print("All sweep runs, worst to best validation loss.")
    print("'100-item' rounds exact16 to what a 100-question test set can resolve (steps of 0.01).\n")
    print(f"{'C':>7s} {'model':9s} {'val loss':>9s} {'exact1':>7s} {'exact4':>7s} {'exact8':>7s} {'exact16':>8s} "
          f"{'exact1^16':>9s} {'100-item':>8s}")
    for r in runs:
        print(f"{r['budget']:7.0e} {r['shape']:9s} {r['val_loss']:9.4f} {r['exact1']:7.4f} {r['exact4']:7.4f} "
              f"{r['exact8']:7.4f} {r['exact16']:8.5f} {r['exact1'] ** 16:9.6f} {round(r['exact16'] * 100) / 100:8.2f}")
    lo, hi = runs[0], runs[-1]
    print(f"\nFrom the worst to the best run, val loss falls {lo['val_loss'] - hi['val_loss']:.3f} nats and next-token "
          f"accuracy rises {lo['exact1']:.3f} -> {hi['exact1']:.3f} (x{hi['exact1'] / lo['exact1']:.2f}).")
    print(f"Exact match on 16 tokens rises {lo['exact16']:.5f} -> {hi['exact16']:.5f} "
          f"(x{hi['exact16'] / max(lo['exact16'], 1e-9):.0f}).")
    print("Per-token accuracy moves by tens of percent; the 16-token score moves by a large factor over the same models,")
    print("because an exact match multiplies many per-token successes. A 100-question test set (last column) scores")
    print("zero for every model here, so it cannot see any of this progress. The sweep spans only about one order of")
    print("magnitude of compute and every model is far from solving 16-token spans, so no sharp transition appears;")
    print("this shows the amplification and the resolution problem, not an emergent jump.")
    print("Measured exact16 is well above exact1^16 because errors cluster: easy spans are easy for every token in them.")


if __name__ == "__main__":
    main()
