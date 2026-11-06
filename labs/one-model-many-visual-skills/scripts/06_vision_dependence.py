"""Does the answer need the image? Blind evaluation and a counterfactual image (post Figure 8, MathVerse, Cambrian-1).

One model is trained on VQA, chat, captions, a small reasoning task with a rationale
("red 2 . blue 1 . 2 minus 1 is 1 .") and 10% text replay. Then:

  image removal    score with a blank image (no objects), as in Cambrian-1's vision-off test.
  image swap       score with an unrelated scene that still holds the object asked about. A suite
                   whose answers come from the text prior (forks are always grey) keeps its score
                   under the swap, so it cannot see any change in visual skill.
  counterfactual   repaint one blue object red and ask the reasoning question again. A rationale
                   that reads the image must change its counts.

    python labs/one-model-many-visual-skills/scripts/06_vision_dependence.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skillmix.runs import sweep, write_result  # noqa: E402
from skillmix.train import RunConfig  # noqa: E402

MIX = {"vqa": 0.25, "chat": 0.15, "caption": 0.15, "reason": 0.35, "text": 0.1}
SUITE = [("eval_text", {})] + [(fn, {**kw, "image": im, "tag": im}) for im in ("real", "blank", "swap")
                               for fn, kw in (("eval_vqa", {"fmt": True}), ("eval_fork_colour", {}), ("eval_reason", {}))]


def main() -> None:
    res = sweep([("mixed model", RunConfig(mixture=MIX, fmt=True, steps=2000), SUITE)])["mixed model"]
    rows = [("VQA (colour, count, position)", "vqa_exact"), ("fork colour (text prior suffices)", "fork_colour_exact"),
            ("reasoning with rationale", "reason_exact")]
    lines = [f"{'suite':<36} {'real':>6} {'blank':>6} {'swap':>6} {'real-swap':>9}"]
    for title, k in rows:
        r, b, w = res[k + "_real"], res[k + "_blank"], res[k + "_swap"]
        lines.append(f"{title:<36} {r:>6.3f} {b:>6.3f} {w:>6.3f} {r - w:>+9.3f}")
    lines += ["", "counterfactual image (one blue object repainted red):",
              f"  rationale exact on the edited image: {res['counterfactual_exact_real']:.3f}",
              f"  rationale identical before and after the edit: {res['rationale_unchanged_real']:.3f}",
              f"  control, image held fixed (swap): unchanged = {res['rationale_unchanged_swap']:.3f}",
              f"text-only suite (reverse): {res['text_exact']:.3f}"]
    text = "\n".join(lines)
    print(text)
    write_result("06_vision_dependence", {"metrics": res}, text)


if __name__ == "__main__":
    main()
