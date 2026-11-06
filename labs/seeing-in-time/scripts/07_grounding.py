"""Finding the moment: temporal IoU, and the ceiling frame spacing puts on it (post Figure 9).

1. The Figure 9 moment (13.4-16.6 s in a 30-second clip): best tIoU reachable at each frame rate.
2. The worst-case floor (L - spacing) / L from the post.
3. A perfect model that can only answer with sampled frame times, on 20,000 random events of 2-10 s:
   R@1 at IoU 0.5 and 0.7, and mIoU, against the frame rate.
4. Coarse-to-fine: sample sparsely to find the moment, then densely around it.
"""

from __future__ import annotations

from _common import Report
from seeingtime.grounding import (best_on_grid, recall_at_1, snap_nearest, snapped_oracle_scores, tiou,
                                  worst_case_floor)

GT = (13.4, 16.6)


def main():
    rep = Report("07_grounding")
    rep.print(f"== 1. Figure 9: true moment {GT[0]}-{GT[1]} s, frames at 0, step, 2*step, ... 30 s ==")
    rows = []
    for fps in (0.25, 0.5, 1, 2, 4):
        b = best_on_grid(GT, 1 / fps, 30)
        rows.append([fps, 1 / fps, f"{b:.3f}", "yes" if b >= 0.7 else "no"])
    rep.table(["fps", "spacing (s)", "best possible tIoU", "can pass IoU 0.7?"], rows)
    rep.print(f"Figure 9's default prediction 12.5-16.0 s: tIoU = {tiou((12.5, 16.0), GT):.3f}")
    rep.data["figure9"] = rows

    rep.print("\n== 2. Worst-case floor (L - spacing) / L ==")
    rows = [[L, d, f"{worst_case_floor(L, d):.3f}"] for L, d in ((4, 2), (4, 0.5), (4, 0.25), (2, 1), (10, 2))]
    rep.table(["event length (s)", "frame spacing (s)", "floor"], rows)
    rep.data["floor"] = rows

    rep.print("\n== 3. Nearest-frame oracle, 20,000 random events of 2-10 s in a 60 s video ==")
    rows = []
    for fps in (0.25, 0.5, 1, 2, 4):
        s = snapped_oracle_scores(1 / fps)
        rows.append([fps, f"{recall_at_1(s, 0.5):.3f}", f"{recall_at_1(s, 0.7):.3f}", f"{sum(s) / len(s):.3f}",
                     f"{min(s):.3f}"])
    rep.table(["fps", "R@1 IoU 0.5", "R@1 IoU 0.7", "mIoU", "worst"], rows)
    rep.print("This is the ceiling for any model that names frame times. Real models sit below it.")
    rep.data["oracle"] = rows

    rep.print("\n== 4. Coarse-to-fine on the Figure 9 moment, 30 s clip ==")
    coarse = snap_nearest(GT, 2.0)
    window = (coarse[0] - 2.0, coarse[1] + 2.0)
    fine = snap_nearest(GT, 0.25)
    frames_coarse = 30 / 2.0 + 1
    frames_fine = (window[1] - window[0]) / 0.25 + 1
    rows = [["0.5 fps everywhere", f"{coarse[0]:.2f}-{coarse[1]:.2f}", f"{tiou(coarse, GT):.3f}", f"{frames_coarse:.0f}"],
            ["0.5 fps, then 4 fps in the window", f"{fine[0]:.2f}-{fine[1]:.2f}", f"{tiou(fine, GT):.3f}",
             f"{frames_coarse + frames_fine:.0f}"],
            ["4 fps everywhere", f"{fine[0]:.2f}-{fine[1]:.2f}", f"{tiou(fine, GT):.3f}", f"{30 * 4 + 1}"]]
    rep.table(["plan", "nearest-frame answer (s)", "tIoU", "frames encoded"], rows)
    rep.print(f"The window is {window[0]:.0f}-{window[1]:.0f} s: the coarse answer widened by one coarse step each side.")
    rep.data["coarse_to_fine"] = rows
    rep.save()


if __name__ == "__main__":
    main()
