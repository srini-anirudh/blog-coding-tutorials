"""Score a saved checkpoint on held-out samples, next to two controls that say what the score is worth.

    python scripts/evaluate.py outputs/stage2_main/trainable.safetensors --name stage2_main --n 64

    seen      the model with the real images
    blind     the same model on grey images: a score that does not drop never came from the image
    majority  no model at all: always the most common training answer
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vlmtrain.c07_adapter import build_c07  # noqa: E402
from vlmtrain.data import TASKS  # noqa: E402
from vlmtrain.device import cap_threads, describe_hardware, pick_device  # noqa: E402
from vlmtrain.evaluate import HEADLINE, evaluate, majority_baseline  # noqa: E402
from vlmtrain.model import load_partial  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--name", required=True)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--tasks", nargs="+", default=list(TASKS))
    args = ap.parse_args()
    cap_threads()
    device = pick_device()
    _, model, proc = build_c07()
    load_partial(model, args.checkpoint)
    model.to(device).eval()
    seen = evaluate(model, proc, device, args.tasks, args.n, keep_examples=5)
    blind = evaluate(model, proc, device, args.tasks, args.n, keep_examples=2, blind=True)
    base = majority_baseline(args.tasks, args.n)
    print(f"{describe_hardware(device)}, n={args.n} per task")
    print(f"{'task':<11}{'metric':<13}{'seen':>8}{'blind':>8}{'majority':>10}")
    for task in args.tasks:
        h = HEADLINE[task]
        print(f"{task:<11}{h:<13}{seen['metrics'][task][h]:>8.3f}{blind['metrics'][task][h]:>8.3f}{base[task][h]:>10.3f}")
    out = {"name": args.name, "hardware": describe_hardware(device), "n": args.n, "seen": seen["metrics"],
           "blind": blind["metrics"], "majority": base, "examples": seen["examples"]}
    (ROOT / "results" / f"eval_{args.name}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
