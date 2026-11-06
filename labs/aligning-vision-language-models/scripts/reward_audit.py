"""Reward models that must look at the image: train two, then run the image-ablation audit (post Figure 4).

    python labs/aligning-vision-language-models/scripts/reward_audit.py --seeds 0 1 2

RM-S is trained on 'style' pairs, where text alone separates winner and loser. RM-G is trained on 'fact' pairs,
where only the image does. Same architecture, same image input. Both are audited on held-out pairs of both kinds.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from groundpref.pairs import build_pairs  # noqa: E402
from groundpref.report import Log  # noqa: E402
from groundpref.reward import audit, train_rm  # noqa: E402
from groundpref.train import set_threads  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--pairs", type=int, default=30000)
    ap.add_argument("--steps", type=int, default=2000)
    args = ap.parse_args()
    set_threads()
    log = Log("reward_audit")
    train = {"RM-S (style pairs)": build_pairs(args.pairs, 10, "style"),
             "RM-G (fact pairs)": build_pairs(args.pairs, 11, "fact")}
    test = {"style": build_pairs(1500, 20, "style"), "fact": build_pairs(1500, 21, "fact")}
    out: dict[str, list] = {}
    for name, pairs in train.items():
        for seed in args.seeds:
            rm = train_rm(pairs, steps=args.steps, seed=seed)
            row = {"seed": seed}
            for tname, tp in test.items():
                a = audit(rm, tp, seed=seed)
                row.update({f"{tname}_{k}": v for k, v in a.items()})
            out.setdefault(name, []).append(row)
            log(f"{name:20s} seed {seed} " + " ".join(f"{k}={v:.3f}" for k, v in row.items() if k != "seed"))
    log("\nmean over seeds")
    keys = ["style_pair_acc", "fact_pair_acc", "fact_pair_acc_object", "fact_pair_acc_colour", "fact_pair_acc_count",
            "fact_sensitivity", "fact_swap_flip"]
    log(f"{'reward model':20s} " + " ".join(f"{k.replace('fact_', ''):>15s}" for k in keys))
    for name, rows in out.items():
        log(f"{name:20s} " + " ".join(f"{statistics.mean(r[k] for r in rows):>15.3f}" for k in keys))
    log.save({"args": vars(args), "results": out})


if __name__ == "__main__":
    main()
