"""The IsoFLOP sweep (sections 6 and 10): at each compute budget, train several model sizes to exactly that budget.

The budget uses Chinchilla's convention, C = 6 * N_total * D. Each run is appended to the output file as one JSON
line as soon as it finishes, so a sweep that stops early keeps what it measured. Fitting is 05_fit_sweep.py.

    python labs/scaling-laws-from-first-principles/scripts/04_sweep.py --plan            # print the grid only
    python labs/scaling-laws-from-first-principles/scripts/04_sweep.py                   # GPU, about 25 minutes
    python labs/scaling-laws-from-first-principles/scripts/04_sweep.py --smoke           # CPU, two tiny runs
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))

import torch  # noqa: E402

from scalelab.data import DEFAULT_DIR, Windows, load_tokens  # noqa: E402
from scalelab.ladder import LADDER  # noqa: E402
from scalelab.train import train_run  # noqa: E402


def grid(budgets: list[float], spread: float, max_tokens: float):
    """Shapes whose N_total is within `spread`x of sqrt(C/120) (20 tokens per parameter), at each budget."""
    runs = []
    for C in budgets:
        guess = math.sqrt(C / 120)
        for s in LADDER:
            D = C / (6 * s.total)
            if guess / spread <= s.total <= guess * spread and D <= max_tokens:
                runs.append((C, s, D))
    return runs


def hardware() -> str:
    if torch.cuda.is_available():
        return f"{torch.cuda.get_device_name(0)}, torch {torch.__version__}, CUDA {torch.version.cuda}, bf16 autocast"
    return f"CPU, {torch.get_num_threads()} threads, torch {torch.__version__}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets", default="1e14,3e14,1e15,3e15")
    ap.add_argument("--spread", type=float, default=6.0)
    ap.add_argument("--out", default=str(LAB / "results" / "sweep.jsonl"))
    ap.add_argument("--data-dir", default=str(DEFAULT_DIR))
    ap.add_argument("--max-minutes", type=float, default=0, help="start no new run after this many minutes")
    ap.add_argument("--plan", action="store_true", help="print the grid and its cost, train nothing")
    ap.add_argument("--smoke", action="store_true", help="two tiny CPU runs on 64k tokens, to check the code path")
    args = ap.parse_args()

    budgets = [float(b) for b in args.budgets.split(",")]
    meta = json.loads((Path(args.data_dir) / "meta.json").read_text())
    available = meta["train"]["tokens"]
    runs = grid(budgets, args.spread, available)
    if args.smoke:
        runs = [(0.0, LADDER[0], 65_536.0), (0.0, LADDER[1], 65_536.0)]
    print(f"{len(runs)} runs; total 6*N_total*D = {sum(c for c, _, _ in runs):.3g} FLOPs; "
          f"train tokens available {available:,}")
    for C, s, D in runs:
        print(f"  C={C:.0e}  {s.name:9s} N_total={s.total / 1e6:6.2f}M  D={D / 1e6:7.1f}M  D/N={D / s.total:6.1f}")
    if args.plan:
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    need = int(max(D for _, _, D in runs)) + 1024
    t0 = time.time()
    train = Windows(load_tokens(Path(args.data_dir), "train", need), LADDER[0].seq, device)
    val = Windows(load_tokens(Path(args.data_dir), "val", 65_536 if args.smoke else 1_048_576), LADDER[0].seq, device)
    print(f"loaded {train.count * train.seq:,} train and {val.count * val.seq:,} val tokens ({time.time() - t0:.0f}s)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    command = f"# command: python {' '.join(sys.argv)}\n".replace(str(LAB), "<lab>")
    if out.exists():  # resume: skip runs already in the file, and log this invocation
        for line in out.read_text().splitlines():
            if line.startswith("{"):
                r = json.loads(line)
                done.add((r["budget"], r["shape"]))
        with out.open("a") as f:
            f.write(f"# hardware: {hardware()}\n" + command)
    else:
        out.write_text(f"# hardware: {hardware()}\n" + command)
    start = time.time()
    for C, s, D in runs:
        if (C, s.name) in done:
            continue
        if args.max_minutes and time.time() - start > 60 * args.max_minutes:
            print(f"stopping: {args.max_minutes} minutes used; rerun the same command to resume")
            break
        r = train_run(s, D, train, val)
        r["budget"] = C
        print(f"C={C:.0e} {s.name:9s} D={r['tokens'] / 1e6:6.1f}M  val {r['val_loss']:.4f}  "
              f"{r['train_seconds']:.0f}s  {r['tokens_per_second'] / 1e6:.2f}M tok/s", flush=True)
        with out.open("a") as f:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
