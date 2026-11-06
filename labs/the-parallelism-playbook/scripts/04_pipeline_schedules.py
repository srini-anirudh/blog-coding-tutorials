"""Pipeline schedules on a clock: GPipe, 1F1B, interleaved 1F1B and a zero-bubble-style schedule.
Checks the bubble formula (P-1)/(m+P-1) and counts live activation stashes on each stage."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ppbook.dist_utils import Report  # noqa: E402
from ppbook.pipeline import ascii_timeline, bubble_formula, simulate  # noqa: E402


def main():
    rep = Report("04_pipeline_schedules.txt")
    rep("Costs: forward 2 units, backward 4 units per stage (backward = 2x forward). Interleaved v=2 runs chunks of")
    rep("half the layers, so 1 + 2 units each. zb splits the backward into B (input grad, 2) and W (weight grad, 2)")
    rep("and fills idle slots with deferred W; it keeps 1F1B's cap of P - s forwards in flight.")
    rep("Stash = one microbatch's activations for one device-stage; an interleaved stash is half that size.")
    rep("")
    rep(f"{'P':>2} {'m':>3} {'formula':>8} {'GPipe':>6} {'1F1B':>6} {'interl.':>8} {'(P-1)/(vm+P-1)':>15} {'zb':>6}"
        f"   peak stashes, stage 0: GPipe / 1F1B / interl. (stage-equiv.) / zb")
    for P in (2, 4, 8):
        for m in (P, 2 * P, 4 * P, 8 * P):
            g, o = simulate(P, m, "gpipe", t_f=2, t_b=4), simulate(P, m, "1f1b", t_f=2, t_b=4)
            i = simulate(P, m, "interleaved", v=2, t_f=2, t_b=4)
            z = simulate(P, m, "zb", t_f=2, t_b=2, t_w=2)
            rep(f"{P:>2} {m:>3} {bubble_formula(P, m):>8.3f} {g['bubble']:>6.3f} {o['bubble']:>6.3f} "
                f"{i['bubble']:>8.3f} {bubble_formula(P, m, 2):>15.3f} {z['bubble']:>6.3f}   "
                f"{g['peak_stashes'][0]} / {o['peak_stashes'][0]} / {i['peak_stashes'][0]} "
                f"({i['peak_stashes'][0] / 2:.1f}) / {z['peak_stashes'][0]}")
    rep("")
    rep("GPipe and 1F1B have the same bubble at every (P, m). 1F1B caps stage 0 at P stashes instead of m.")
    rep(f"P=8, m=8 wastes {bubble_formula(8, 8):.0%}; P=8, m=64 wastes {bubble_formula(8, 64):.0%} (the post's 47% and 10%).")
    rep("")
    for name, kw in (("GPipe", dict(schedule="gpipe")), ("1F1B", dict(schedule="1f1b")),
                     ("interleaved, v=2 (lower case = second chunk)", dict(schedule="interleaved", v=2)),
                     ("zb (w = deferred weight gradient)", dict(schedule="zb", t_b=2, t_w=2))):
        kw = {"t_f": 2, "t_b": 4, **kw}
        sim = simulate(4, 8, **kw)
        rep(f"P=4, m=8, {name}: bubble {sim['bubble']:.3f}, makespan {sim['makespan']} units. "
            "Each op starts with its microbatch number" + (", 2 characters per unit." if kw.get("v") == 2 else "."))
        rep(ascii_timeline(sim, 2 if kw.get("v") == 2 else 1))
        rep("")
    rep.save()


if __name__ == "__main__":
    main()
