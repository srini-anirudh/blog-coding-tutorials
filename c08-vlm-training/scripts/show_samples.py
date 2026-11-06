"""Save one rendered sample per task as a PNG and print its unified dict.

    python c08-vlm-training/scripts/show_samples.py --out outputs/samples
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vlmtrain.data import TASKS, make_example  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/samples")
    ap.add_argument("--index", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for task in TASKS:
        s = make_example(task, args.index)
        s["images"][0].save(out / f"{task}.png")
        q = s["messages"][0]["content"][-1]["text"]
        print(f"{task:<10} {s['images'][0].size}  Q: {q}\n{'':<10} A: {s['meta']['answer']}")


if __name__ == "__main__":
    main()
