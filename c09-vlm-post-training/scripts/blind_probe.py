"""Can a model that never sees the image predict which answer was chosen?

    python scripts/blind_probe.py --train 1000 --test 500
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from vlmalign.blind import fit_text_probe  # noqa: E402
from vlmalign.preference import build_pairs  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=1000)
    ap.add_argument("--test", type=int, default=500)
    args = ap.parse_args()
    torch.set_num_threads(2)
    results = {}
    print(f"{'negatives':10s} {'train':>7s} {'test':>7s}   per kind (test)")
    for mode in ("balanced", "biased"):
        r = fit_text_probe(build_pairs(args.train, mode, seed=21), build_pairs(args.test, mode, seed=22))
        kinds = "  ".join(f"{k}={v:.2f}" for k, v in r.per_kind.items())
        print(f"{mode:10s} {r.train_acc:7.3f} {r.test_acc:7.3f}   {kinds}")
        results[mode] = {"train": r.train_acc, "test": r.test_acc, "per_kind": r.per_kind}
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / "blind_probe.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
