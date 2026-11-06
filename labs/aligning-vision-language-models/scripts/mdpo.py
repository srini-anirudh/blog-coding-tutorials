"""Multimodal DPO and the ignored image: standard DPO, the no-image ablation, and mDPO.

    python labs/aligning-vision-language-models/scripts/mdpo.py --seeds 0 1 2

Every variant starts from the same SFT policy and trains on the same style-confounded pairs (the winner is plain
and correct, the loser has one error and style words). We then ask three things:
  1. Hallucination when describing held-out scenes (greedy decoding, exact truth).
  2. Accuracy on 'fact' pairs that only the image can separate.
  3. Where the training margin came from: beta * D_text versus beta * D_image.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from groundpref.pairs import build_pairs  # noqa: E402
from groundpref.prefopt import VARIANTS, fact_accuracy, split_on, train_pref  # noqa: E402
from groundpref.report import Log  # noqa: E402
from groundpref.train import describe_eval, eval_scenes, load_sft, set_threads  # noqa: E402

COLS = ["chair_i", "recall", "colour_err", "count_err", "policy_pref_acc", "implicit_reward_acc",
        "image_gain_yw", "beta_d_text", "beta_d_image"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--pairs", type=int, default=4000)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--name", default="mdpo")
    args = ap.parse_args()
    set_threads()
    log = Log(args.name)
    ref = load_sft(log)
    train = build_pairs(args.pairs, seed=1000, flavour="style")
    fact = build_pairs(1500, seed=2000, flavour="fact")
    scenes, imgs = eval_scenes(600, seed=3000)
    split_pairs = build_pairs(600, seed=1000, flavour="style")  # same seed: the first 600 training pairs

    base = {**describe_eval(ref, scenes, imgs), **fact_accuracy(ref, ref, fact)}
    log("sft  " + "  ".join(f"{k}={base[k]:.3f}" for k in COLS if k in base))
    runs: dict[str, list[dict]] = {}
    for name in args.variants:
        for seed in args.seeds:
            pol, curve = train_pref(ref, train, VARIANTS[name], steps=args.steps, seed=seed, log=lambda *a: None)
            r = {**describe_eval(pol, scenes, imgs), **fact_accuracy(pol, ref, fact), **split_on(pol, ref, split_pairs),
                 "final_dpo_loss": curve[-1]["dpo_loss"], "seed": seed}
            runs.setdefault(name, []).append(r)
            log(f"{name:16s} seed {seed} dpo_loss={r['final_dpo_loss']:.4f} " +
                "  ".join(f"{k}={r[k]:.3f}" for k in COLS))

    log("\nmean over seeds (sd)")
    log(f"{'variant':16s} " + " ".join(f"{k:>18s}" for k in ["final_dpo_loss"] + COLS))
    summary = {"sft": base}
    for name, rs in runs.items():
        m = {k: statistics.mean(r[k] for r in rs) for k in ["final_dpo_loss"] + COLS}
        sd = {k: (statistics.stdev(r[k] for r in rs) if len(rs) > 1 else 0.0) for k in m}
        summary[name] = {"mean": m, "sd": sd, "runs": rs}
        log(f"{name:16s} " + " ".join(f"{m[k]:>11.3f} ({sd[k]:.3f})" for k in m))
    log.save({"args": vars(args), "results": summary})


if __name__ == "__main__":
    main()
