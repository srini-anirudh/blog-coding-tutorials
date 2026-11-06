"""Short answers against conversation, and the format prompt that resolves the conflict (post Figure 4).

The VQA and chat sources ask the same questions. VQA answers in one word or phrase. Chat answers in
two sentences. We sweep the VQA share. Without a format prompt the two styles share one input
format, so the model must average them. With LLaVA-1.5's fix, every VQA question ends with
"answer with a word or phrase .", and the style becomes a function of the input.

Evaluation asks unmarked questions, as a user would, and counts long answers (6 words or more).

    python labs/one-model-many-visual-skills/scripts/03_format_conflict.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skillmix.runs import sweep, write_result  # noqa: E402
from skillmix.train import RunConfig  # noqa: E402

SHARES = (0.0, 0.25, 0.5, 0.75, 0.9)


def main() -> None:
    jobs = []
    for fmt in (False, True):
        for s in SHARES:
            suite = [("eval_chat", {}), ("eval_vqa", {"fmt": fmt})]
            jobs.append((f"fmt={'on' if fmt else 'off'} vqa={s}", RunConfig(mixture={"vqa": s, "chat": 1 - s}, fmt=fmt), suite))
    res = sweep(jobs)
    lines = ["Unmarked questions after training. long = answers of 6+ words. correct = right value in the answer.",
             "VQA exact is scored with the prompt format used in training (with the format prompt only when it is on).",
             "", f"{'run':<20} {'long rate':>9} {'mean words':>10} {'correct':>8} {'VQA exact':>9}"]
    for name, r in res.items():
        lines.append(f"{name:<20} {r['chat_long_rate']:>9.3f} {r['chat_mean_words']:>10.2f} "
                     f"{r['chat_correct']:>8.3f} {r['vqa_exact']:>9.3f}")
    text = "\n".join(lines)
    print(text)
    write_result("03_format_conflict", {"runs": res}, text)


if __name__ == "__main__":
    main()
