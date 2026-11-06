"""How a teacher's hallucination reaches the student, and how verification stops it (post Figure 6).

In this world a fork sits next to a plate 70% of the time. The biased teacher knows that, and when a
plate has no fork it adds "there is a grey fork at ..." as the last sentence, 90% of the time.
The bias is built in by hand, late in the caption, to mirror what POPE and LURE report. It is not a
reproduction of those papers. The question is whether the student copies it.

Students: captions from the clean annotator; from the biased teacher; from the teacher after an
object-level check that drops sentences naming absent objects. All three also see VQA, chat and
balanced "is there a ... ?" questions. Evaluation uses scenes with a plate and no fork.

    python labs/one-model-many-visual-skills/scripts/05_teacher_hallucination.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skillmix.runs import sweep, write_result  # noqa: E402
from skillmix.train import RunConfig  # noqa: E402

MIX = {"caption": 0.4, "vqa": 0.2, "chat": 0.2, "exist": 0.2}
SUITE = [("eval_caption", {}), ("eval_exist", {})]


def main() -> None:
    runs = [("clean annotator", 0.0, False), ("biased teacher", 0.9, False), ("teacher + verification", 0.9, True)]
    res = sweep([(n, RunConfig(mixture=MIX, fmt=True, teacher_bias=b, verify=v), SUITE) for n, b, v in runs])
    lines = ["Eval scenes: a plate and no fork. fork rate = captions that mention a fork.",
             "CHAIR_i = mentioned objects not in the scene / all mentioned. yes rate = 'is there a fork ?' -> yes.", "",
             f"{'student':<24} {'fork rate':>9} {'CHAIR_i':>8} {'recall':>7} {'fork yes rate':>13}"]
    for name, r in res.items():
        lines.append(f"{name:<24} {r['fork_rate']:>9.3f} {r['chair_i']:>8.3f} {r['recall']:>7.3f} {r['fork_yes_rate']:>13.3f}")
    lines += ["", "one caption from each student:"] + [f"  {n}: {r['sample']}" for n, r in res.items()]
    text = "\n".join(lines)
    print(text)
    write_result("05_teacher_hallucination", {"runs": res}, text)


if __name__ == "__main__":
    main()
