"""Per-token vs per-example loss averaging on a 50/50 mix of one-word VQA and long captions.

Under per-token averaging the short answers get a small share of the signal (script 01): the
normalization silently changes the mixture. Which skill gains or loses from that is measured here,
with a short budget (one seed) and the full budget (two seeds).

    python labs/one-model-many-visual-skills/scripts/02_loss_normalization.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skillmix.runs import sweep, write_result  # noqa: E402
from skillmix.train import RunConfig  # noqa: E402

SUITE = [("eval_vqa", {"fmt": False}), ("eval_caption", {})]


def main() -> None:
    jobs = [(f"{norm}_{steps}_s{seed}", RunConfig(mixture={"vqa": 0.5, "caption": 0.5}, norm=norm, steps=steps,
                                                  seed=seed), SUITE)
            for steps, seed in ((250, 0), (1000, 0), (1000, 1)) for norm in ("token", "example")]
    res = sweep(jobs)
    lines = ["mix: 50% one-word VQA, 50% detailed captions (examples). Within a seed the data stream is identical; only the averaging differs.",
             "", f"{'run':<18} {'VQA exact':>9} {'caption recall':>15} {'CHAIR_i':>8}"]
    for name, r in res.items():
        lines.append(f"{name:<18} {r['vqa_exact']:>9.3f} {r['recall']:>15.3f} {r['chair_i']:>8.3f}")
    text = "\n".join(lines)
    print(text)
    write_result("02_loss_normalization", {"runs": res}, text)


if __name__ == "__main__":
    main()
