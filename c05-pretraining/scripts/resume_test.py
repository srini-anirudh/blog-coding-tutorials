"""Exact resume on real data: uninterrupted, interrupted and resumed, and a weights-only restart.

    python c05-pretraining/scripts/resume_test.py --steps 40
    python c05-pretraining/scripts/resume_test.py --steps 40 --model tinyllm
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from pretrain.config import TrainConfig  # noqa: E402
from pretrain.trainer import train  # noqa: E402
from pretrain.utils import resolve  # noqa: E402


def losses(records: list[dict]) -> dict[int, float]:
    return {r["step"]: r["loss"] for r in records if "loss" in r}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--model", default="llama", choices=["llama", "tinyllm"])
    ap.add_argument("--out", help="where to write the JSON (default: results/resume[_<model>].json)")
    args = ap.parse_args()
    half = args.steps // 2
    base = TrainConfig.from_yaml(resolve("configs/smoke.yaml"), max_steps=args.steps, log_every=1,
                                 ckpt_every=half, eval_every=10_000, device=args.device,
                                 warmup_steps=5, out_dir="runs/resume/straight", model_type=args.model)
    root = resolve("runs/resume")
    shutil.rmtree(root, ignore_errors=True)

    straight = losses(train(base, resume=False))

    base.out_dir = "runs/resume/split"
    resumed = losses(train(base, resume=False, stop_after=half))
    resumed |= losses(train(base, resume=True))

    # The usual mistake: reload the weights and start everything else fresh.
    base.out_dir = "runs/resume/weights_only"
    wrong = losses(train(base, resume=False, stop_after=half))
    ckpt = next(root.glob("weights_only/ckpt_*.pt"))
    state = torch.load(ckpt, weights_only=True)
    keep = {"model": state["model"], "config": state["config"]}
    fresh = TrainConfig.from_dict(state["config"])
    fresh.out_dir = "runs/resume/weights_only_2"
    wrong |= {half + k: v for k, v in losses(_train_from_weights(fresh, keep["model"], half)).items()}

    print(f"model {args.model}, device {args.device}, {args.steps} steps, interrupted at step {half}")
    print(f"{'step':>5} {'straight':>10} {'resumed':>10} {'weights only':>13}")
    for s in [half - 1, half, half + 1, half + 2, half + 3, half + 5, half + 10, args.steps]:
        if s in straight:
            print(f"{s:>5} {straight[s]:>10.6f} {resumed[s]:>10.6f} {wrong[s]:>13.6f}")
    same = all(straight[s] == resumed[s] for s in straight)
    gap = max(abs(straight[s] - resumed[s]) for s in straight)
    print(f"bitwise equal: {same}  (max |straight - resumed| = {gap:.2e})")
    default = "results/resume.json" if args.model == "llama" else f"results/resume_{args.model}.json"
    out = resolve(args.out or default)
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"straight": straight, "resumed": resumed, "weights_only": wrong, "half": half}))
    shutil.rmtree(root, ignore_errors=True)


def _train_from_weights(cfg: TrainConfig, weights: dict, steps: int) -> list[dict]:
    """Weights restored; optimizer, schedule and data position all restart from zero."""
    from pretrain import trainer

    original = trainer.build_model

    def build_with_weights(a, t):
        model = original(a, t)
        model.load_state_dict(weights)
        return model

    trainer.build_model = build_with_weights
    try:
        return train(cfg, resume=False, stop_after=steps)
    finally:
        trainer.build_model = original


if __name__ == "__main__":
    main()
