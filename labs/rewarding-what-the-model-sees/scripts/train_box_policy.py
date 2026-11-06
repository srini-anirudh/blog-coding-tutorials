"""Train the box-size policy with GRPO under each box reward and report how large the boxes get.

    python labs/rewarding-what-the-model-sees/scripts/train_box_policy.py --steps 300 --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from visrl.boxes import DIFFICULTIES, REWARDS, train_box_policy  # noqa: E402
from visrl.report import table, write  # noqa: E402


def spread(xs: list[float], digits: int = 2) -> str:
    m = sum(xs) / len(xs)
    return f"{m:.{digits}f} [{min(xs):.{digits}f}, {max(xs):.{digits}f}]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()
    t0 = time.time()
    configs = [(r, True) for r in REWARDS] + [("iou", False)]
    rows, data = [], {}
    for reward, scale in configs:
        runs = [train_box_policy(reward, steps=args.steps, seed=s, scale_by_std=scale) for s in args.seeds]
        name = reward + ("" if scale else " (no std)")
        last = lambda xs: sum(xs[-50:]) / 50  # noqa: E731
        rows.append([name,
                     spread([r.area_ratio["easy"][-1] for r in runs]),
                     spread([r.area_ratio["hard"][-1] for r in runs]),
                     spread([last(r.useful["hard"]) for r in runs]),
                     spread([r.final_iou["hard"] for r in runs], 3)])
        data[name] = [{"seed": s, "area_ratio": {d: r.area_ratio[d][::25] + [r.area_ratio[d][-1]] for d in DIFFICULTIES},
                       "useful_last50": {d: last(r.useful[d]) for d in DIFFICULTIES}, "final_iou": r.final_iou}
                      for s, r in zip(args.seeds, runs)]
    text = (f"Box-size policy, GRPO, K = 8, 64 targets per difficulty per step, {args.steps} steps, "
            f"seeds {args.seeds}. Location noise: easy sigma = {DIFFICULTIES['easy']}, hard sigma = "
            f"{DIFFICULTIES['hard']} image widths; targets have sides 0.15-0.35. Start: boxes the target's size "
            "(ratio 1.00). Mean [min, max] over seeds.\n\n"
            + table(rows, ["reward", "box/target area, easy", "box/target area, hard",
                           "useful groups, hard (last 50 steps)", "mean IoU, hard"])
            + f"\n\nwall time: {time.time() - t0:.0f} s")
    write("box_policy", text, {"steps": args.steps, "seeds": args.seeds, "runs": data})


if __name__ == "__main__":
    torch.set_num_threads(min(torch.get_num_threads(), 8))
    main()
