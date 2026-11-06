"""Score the cheapest degenerate output of each reward against an honest miss.

The post's checklist, item one: before training, compute the reward of the whole
image, an empty answer, every category listed and a hedged answer. If the cheap
output beats a typical miss, fix the reward first. No model is needed.

    python labs/rewarding-what-the-model-sees/scripts/audit_rewards.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from visrl import verifiers as V  # noqa: E402
from visrl.report import table, write  # noqa: E402

TARGET = (0.25, 0.30, 0.75, 0.70)        # covers 20% of the unit image, as in the post's example
WHOLE = (0.0, 0.0, 1.0, 1.0)
MISS = (0.0, 0.0, 0.5, 0.3)              # a precise-looking box, 0.75x the target area, just above it


def boxes() -> list[list]:
    rewards = {
        "soft IoU": V.box_iou, "IoU >= 0.5": V.iou_at, "GIoU": V.giou, "coverage": V.coverage,
        "centre hit": V.center_hit, "IoU + size": lambda p, g: 0.5 * V.box_iou(p, g) + 0.5 * V.size_match(p, g),
    }
    rows = []
    for name, fn in rewards.items():
        cheap, miss = fn(WHOLE, TARGET), fn(MISS, TARGET)
        rows.append([name, "whole-image box", f"{cheap:.3f}", f"{miss:.3f}", "yes" if cheap > miss else "no"])
    return rows


def others() -> list[list]:
    gold_json = {"vendor": "Acme Ltd", "date": "2024-03-01", "total": "41.20"}
    schema = {"vendor", "date", "total", "tax", "currency", "invoice_no", "address", "phone"}
    honest_json = '{"vendor": "Acme Ltd", "total": "41.20"}'   # leaves out the date it cannot read
    fill_all = ('{"vendor": "Acme Ltd", "date": "2024-03-01", "total": "41.20", "tax": "0", "currency": "USD", '
                '"invoice_no": "1", "address": "Main St", "phone": "000"}')  # guesses every field
    gold_det = {"dog": [(0.1, 0.1, 0.4, 0.5)]}
    honest_det = {"dog": [(0.1, 0.1, 0.4, 0.5)], "cat": [(0.6, 0.6, 0.9, 0.9)]}   # one false positive
    every_cat = {c: [(0.1, 0.1, 0.4, 0.5)] for c in ["dog", "cat", "car", "bus", "cup", "tv", "bed", "pen"]}
    n_pred = lambda d: sum(map(len, d.values()))  # noqa: E731
    rows = [
        ["multiple choice, lenient parser", "'B or C' (gold C)", V.lenient_choice("B or C", "C"),
         V.lenient_choice("<answer>B</answer>", "C")],
        ["multiple choice, strict parser", "'B or C' (gold C)", V.strict_choice("B or C", "C"),
         V.strict_choice("<answer>B</answer>", "C")],
        ["numeric 5%, first number", "'between 56 and 60' (gold 58)",
         V.relaxed_accuracy("between 56 and 60", "58", strict=False), V.relaxed_accuracy("61", "58", strict=False)],
        ["numeric 5%, one number only", "'between 56 and 60' (gold 58)",
         V.relaxed_accuracy("between 56 and 60", "58"), V.relaxed_accuracy("61", "58")],
        ["category score (COCO-style)", "box for all 8 listed categories",
         V.category_score(every_cat, gold_det), V.category_score(honest_det, gold_det)],
        ["category score, length-aware", "box for all 8 listed categories",
         V.length_aware(V.category_score(every_cat, gold_det), n_pred(every_cat), 1),
         V.length_aware(V.category_score(honest_det, gold_det), n_pred(honest_det), 1)],
        ["JSON field recall", "every schema field guessed", V.json_field_recall(fill_all, gold_json, schema),
         V.json_field_recall(honest_json, gold_json, schema)],
        ["JSON schema gate x F1", "every schema field guessed", V.json_field_f1(fill_all, gold_json, schema),
         V.json_field_f1(honest_json, gold_json, schema)],
        ["ANLS", "empty string", V.anls("", "TOTAL 41.20"), V.anls("T0TAL 41.2O", "TOTAL 41.20")],
    ]
    out = []
    for name, cheap_out, cheap, miss in rows:
        out.append([name, cheap_out, f"{cheap:.3f}", f"{miss:.3f}", "yes" if cheap > miss else "no"])
    return out


def facts() -> list[str]:
    """The arithmetic claims of the box section, checked."""
    inside2 = (0.25, 0.12, 0.75, 0.88)               # contains the target, 1.9 times its area
    inside21 = (0.25, 0.08, 0.75, 0.92)              # contains it, 2.1 times its area
    near, far = (0.76, 0.30, 1.00, 0.70), (0.00, 0.00, 0.24, 0.10)
    return [
        f"whole-image box vs a 20% target: IoU = {V.box_iou(WHOLE, TARGET):.3f}, GIoU = {V.giou(WHOLE, TARGET):.3f}, "
        f"coverage = {V.coverage(WHOLE, TARGET):.3f}",
        f"containing box with 1.9x the target area: IoU = {V.box_iou(inside2, TARGET):.3f}, passes IoU>=0.5: "
        f"{bool(V.iou_at(inside2, TARGET))}; with 2.1x: IoU = {V.box_iou(inside21, TARGET):.3f}, passes: "
        f"{bool(V.iou_at(inside21, TARGET))}",
        f"GIoU ranks misses that IoU cannot: near miss {V.giou(near, TARGET):.3f}, far miss "
        f"{V.giou(far, TARGET):.3f} (IoU {V.box_iou(near, TARGET):.3f} for both)",
        f"GIoU equals IoU for any containing box: 1.9x box GIoU = {V.giou(inside2, TARGET):.3f}",
        f"relaxed accuracy: 57.8 vs 58 -> {V.relaxed_accuracy('57.8', '58'):.0f}; a misreading 56 vs 58 -> "
        f"{V.relaxed_accuracy('56', '58'):.0f} (inside the 5% band); 55 vs 58 -> {V.relaxed_accuracy('55', '58'):.0f}",
    ]


def main() -> None:
    head = ["reward", "cheap output", "cheap reward", "honest miss reward", "cheap beats miss?"]
    text = ("Box rewards. Target covers 20% of the image; the honest miss is a box of similar size (0.75x) placed just above it.\n\n"
            + table(boxes(), head)
            + "\n\nOther rewards. The honest miss is a wrong or partial answer a real model might give.\n\n"
            + table(others(), head)
            + "\n\nChecks of the box-section arithmetic:\n" + "\n".join("- " + f for f in facts()))
    write("audit_rewards", text)


if __name__ == "__main__":
    main()
