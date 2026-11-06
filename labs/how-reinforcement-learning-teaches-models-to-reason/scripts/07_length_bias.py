"""Part of the length curve is a normalization bug: wrong answers grow under GRPO's 1/|o_i|.

    python labs/how-reinforcement-learning-teaches-models-to-reason/scripts/07_length_bias.py
    python labs/how-reinforcement-learning-teaches-models-to-reason/scripts/07_length_bias.py --seeds 1 --steps 100
"""

import argparse

import _path  # noqa: F401

from rlvrlab.common import Report
from rlvrlab.length_bias import LOSSES, LengthConfig, stop_gradient_bias, train


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    rep = Report("07_length_bias")
    rep.print("Toy: hidden state 'knows' (right 90%) or 'guesses' (right 10%), seen by the policy. Length never")
    rep.print("affects correctness. Start: mean length 40 in both states. 64 prompts x 8 samples per step, Adam lr 0.02.")
    rep.print()
    rep.print("1. Estimator bias at initialization, no optimizer: mean push on each state's stop logit")
    rep.print("   (negative = write longer), over 200000 groups.")
    brows, bias = [], {}
    for loss in LOSSES:
        b = stop_gradient_bias(LengthConfig(loss=loss))
        bias[loss] = b
        brows.append([loss, b["guess"], b["guess_stderr"], b["guess"] / b["guess_stderr"], b["knows"],
                      b["knows"] / b["knows_stderr"]])
    rep.table(["loss", "push (guess)", "stderr", "z (guess)", "push (knows)", "z (knows)"], brows)
    rep.print()

    rep.print(f"2. Training, {args.steps} steps, {args.seeds} seeds. Lengths are means over the step's samples.")
    runs, rows = {}, []
    for loss in LOSSES:
        runs[loss] = [train(LengthConfig(loss=loss, steps=args.steps, seed=s)) for s in range(args.seeds)]
        for s, log in enumerate(runs[loss]):
            first, last = log[0], log[-1]
            rows.append([loss, s, first["len_wrong"], last["len_wrong"], last["len_right"], last["len_guess"],
                         last["len_knows"], last["accuracy"], last["truncated"]])
    rep.table(["loss", "seed", "wrong len @0", f"wrong len @{args.steps}", "right len", "E[len] guess",
               "E[len] knows", "accuracy", "truncated"], rows)
    rep.print()
    rep.print("Mean length of wrong responses over training (seed 0):")
    steps = [r["step"] for r in runs[LOSSES[0]][0]][:: max(1, args.steps // 80)]
    trow = [[st] + [next(r["len_wrong"] for r in runs[l][0] if r["step"] == st) for l in LOSSES] for st in steps]
    rep.table(["step"] + list(LOSSES), trow)
    rep.data = {"bias": bias, "final": rows, "curves": {l: runs[l][0] for l in LOSSES}}
    rep.save()


if __name__ == "__main__":
    main()
