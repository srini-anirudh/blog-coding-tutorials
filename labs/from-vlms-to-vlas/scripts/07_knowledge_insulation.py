"""Knowledge insulation, as a toy: train a new action expert with and without a stop-gradient into the backbone.

See vlalab/insulation.py for the task. The question: after robot training on 5 colours, does the policy still follow
instructions that name the 15 colours only the pretraining saw?
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vlalab.insulation import (ToyVLA, chance_error, evaluate, pretrain, robot_train,  # noqa: E402
                               wrong_object_error)
from vlalab.report import Report  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrain-steps", type=int, default=2000)
    ap.add_argument("--robot-steps", type=int, default=1500)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "results" / "07_knowledge_insulation"))
    args = ap.parse_args()
    rep = Report(args.out, " ".join(["scripts/07_knowledge_insulation.py"] + sys.argv[1:]))
    rep.line(f"Pretraining (pointing, 20 colours): {args.pretrain_steps} steps. Robot training (flow matching, "
             f"colours 0-4): {args.robot_steps} steps. Errors are mean distances; positions lie in [-1, 1]^2.")
    rep.line(f"Reference errors: chunk that does not move {chance_error():.3f}; going to a random other object "
             f"{wrong_object_error():.3f}.")
    variants = ["joint", "joint+web", "insulated", "frozen"]
    per_seed = []
    for seed in range(args.seeds):
        torch.manual_seed(seed)
        base = ToyVLA()
        pretrain(base, args.pretrain_steps, seed=seed)
        row = {"seed": seed, "pretrained pointing_err": evaluate(base)["pointing_err_all"]}
        for v in variants:
            row[v] = evaluate(robot_train(base, v, args.robot_steps, seed=seed))
        per_seed.append(row)
        rep.line(f"seed {seed}: pretrained pointing error {row['pretrained pointing_err']}")
    rows = []
    for v in variants:
        r = {"variant": v}
        for k in per_seed[0][v]:
            vals = [s[v][k] for s in per_seed]
            r[k] = f"{sum(vals) / len(vals):.3f} ({min(vals):.3f}-{max(vals):.3f})"
        rows.append(r)
    rep.line(f"\nMean over {args.seeds} seeds (min-max):")
    rep.table(rows)
    rep.save_json({"per_seed": per_seed, "summary": rows})


if __name__ == "__main__":
    main()
