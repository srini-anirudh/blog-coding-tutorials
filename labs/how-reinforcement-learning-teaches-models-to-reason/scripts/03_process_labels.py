"""Outcome and process: lucky trajectories, and Monte Carlo step labels that measure the completer.

    python labs/how-reinforcement-learning-teaches-models-to-reason/scripts/03_process_labels.py
"""

import _path  # noqa: F401

from rlvrlab.common import Report
from rlvrlab.prm import label_accuracy, mc_step_labels, outcome_false_positive_rate, sample_derivations

COMPLETERS = {  # (a_clean, a_rescue): chance of reaching the right answer from a clean / broken prefix
    "weak completer": (0.5, 0.02),
    "medium completer": (0.8, 0.15),
    "strong completer": (0.95, 0.4),
}


def main() -> None:
    rep = Report("03_process_labels")
    step_ok, answer_ok = sample_derivations(100_000, steps=6, p_err=0.1, p_lucky=0.15)
    fp = outcome_false_positive_rate(step_ok, answer_ok)
    rep.print("100000 six-step derivations, first-mistake rate 0.1 per step, 0.15 chance a broken one lands right.")
    rep.print(f"Outcome reward 1 rate: {answer_ok.mean():.3f}")
    rep.print(f"Rewarded trajectories that contain a wrong step: {fp:.3f}")
    rep.print()
    rows = []
    for name, (a_clean, a_rescue) in COMPLETERS.items():
        hard, soft = mc_step_labels(step_ok, a_clean, a_rescue, m=8)
        h, s = label_accuracy(step_ok, hard), label_accuracy(step_ok, soft)
        rows.append([name, a_clean, a_rescue, h["good_steps_called_good"], h["bad_steps_called_bad"],
                     s["good_steps_called_good"], s["bad_steps_called_bad"]])
    rep.print("Monte Carlo step labels, 8 completions per step. hard = any completion right; soft = more than half.")
    rep.table(["completer", "a_clean", "a_rescue", "hard: good->good", "hard: bad->bad", "soft: good->good",
               "soft: bad->bad"], rows)
    rep.data = {"outcome_reward_rate": float(answer_ok.mean()), "false_positive_rate": fp, "labels": rows}
    rep.save()


if __name__ == "__main__":
    main()
