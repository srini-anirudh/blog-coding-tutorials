"""Score hand-written answers with every reward, including the weak ones.

No model here. If an obviously bad answer earns a high score, the reward is
broken before training starts, and this table costs nothing to print.

    python scripts/reward_audit.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vlmalign import rewards as R  # noqa: E402
from vlmalign.rl_tasks import COLOR_VOCAB  # noqa: E402


def main() -> None:
    gold_box = (120, 200, 220, 300)
    print("task: box (gold = 120 200 220 300 on a 512 canvas)")
    print(f"  {'answer':26s} {'center_in_box':>14s} {'iou':>7s}")
    for text in ("120 200 220 300", "130 210 230 310", "0 0 512 512", "the box is here"):
        box = R.parse_box(text)
        weak = 0.0 if box is None else R.center_in_box(box, gold_box)
        print(f"  {text:26s} {weak:14.2f} {R.box_reward(text, gold_box):7.3f}")

    truth = {"red", "blue"}
    print("\ntask: list colours (truth = red, blue)")
    print(f"  {'answer':34s} {'recall':>7s} {'f1':>7s}")
    for text in ("red and blue", "red", "red, blue, green, yellow, purple", "blue, green"):
        print(f"  {text:34s} {R.caption_recall(text, truth, COLOR_VOCAB):7.2f} "
              f"{R.caption_f1(text, truth, COLOR_VOCAB):7.2f}")

    print("\ntask: count (truth = 3)")
    print(f"  {'answer':26s} {'contains':>9s} {'exact':>7s}")
    for text in ("3", "three.", "1 2 3 4 5", "4"):
        print(f"  {text:26s} {R.contains_match(text, '3'):9.2f} {R.exact_match(text, '3'):7.2f}")


if __name__ == "__main__":
    main()
