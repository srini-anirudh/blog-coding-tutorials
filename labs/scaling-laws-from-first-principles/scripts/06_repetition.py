"""Section 9: repeated tokens are worth less than fresh ones. One model, one compute budget, fewer unique tokens.

Every run trains d192-L6 (3.4M parameters) on the same 48.4M training tokens (C = 1e15 FLOPs), drawn from only the
first 48.4M / R unique tokens, for R = 1, 2, 4, ... epochs. Compute is identical, so any loss gap is the price of
repetition. `--analyze` compares the measured losses with Muennighoff et al.'s effective-token formula, using the
loss law that 05_fit_sweep.py fitted to this lab's own sweep.

    python labs/scaling-laws-from-first-principles/scripts/06_repetition.py              # GPU, about 4 minutes
    python labs/scaling-laws-from-first-principles/scripts/06_repetition.py --analyze    # CPU, reads the results
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))

import torch  # noqa: E402

from scalelab.allocation import LossLaw, effective_tokens  # noqa: E402
from scalelab.data import DEFAULT_DIR, Windows, load_tokens  # noqa: E402
from scalelab.ladder import LADDER  # noqa: E402
from scalelab.train import train_run  # noqa: E402

SHAPE = LADDER[4]  # d192-L6
TOKENS = 1e15 / (6 * SHAPE.total)
EPOCHS = [1, 2, 4, 8, 16, 32, 64]
OUT = LAB / "results" / "repetition.jsonl"


def read(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.startswith("{")]


def analyze() -> None:
    runs = sorted(read(OUT), key=lambda r: r["epochs"])
    law_path = LAB / "results" / "fit_sweep.json"
    law = LossLaw(**json.loads(law_path.read_text())["parametric_total"]) if law_path.exists() else None
    base = runs[0]["val_loss"]
    hdr = OUT.read_text().splitlines()[0]
    print(hdr.replace("# hardware:", "# training hardware:"))
    print(f"# model {SHAPE.name}, N_total {SHAPE.total:,}, training tokens {runs[0]['tokens']:,} in every run\n")
    print(f"{'epochs':>6} {'unique':>9} {'val loss':>9} {'vs 1 epoch':>10} {'D_eff (R*=15.4)':>16} {'law at D_eff':>12}")
    for r in runs:
        u, ep = r["unique_tokens"], r["epochs"]
        deff = effective_tokens(u, ep)
        pred = f"{float(law(SHAPE.total, deff)):.4f}" if law else "-"
        print(f"{ep:6.1f} {u / 1e6:8.2f}M {r['val_loss']:9.4f} {100 * (r['val_loss'] / base - 1):+9.2f}% "
              f"{deff / 1e6:15.1f}M {pred:>12}")
    if law:
        print(f"\nlaw: the parametric fit to the 6*N_total sweep (results/fit_sweep.json), evaluated at N = N_total"
              f" and D = D_eff.\nAt 1 epoch the law predicts {float(law(SHAPE.total, runs[0]['tokens'])):.4f}.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--data-dir", default=str(DEFAULT_DIR))
    args = ap.parse_args()
    if args.analyze:
        return analyze()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = Windows(load_tokens(Path(args.data_dir), "train", int(TOKENS) + 1024), SHAPE.seq, device)
    val = Windows(load_tokens(Path(args.data_dir), "val", 1_048_576), SHAPE.seq, device)
    done = {round(r["epochs"]) for r in read(OUT)} if OUT.exists() else set()
    if not OUT.exists():
        hw = f"{torch.cuda.get_device_name(0)}, torch {torch.__version__}, CUDA {torch.version.cuda}, bf16 autocast" \
            if device.type == "cuda" else f"CPU, torch {torch.__version__}"
        OUT.write_text(f"# hardware: {hw}\n# command: python scripts/06_repetition.py\n")
    for ep in EPOCHS:
        if ep in done:
            continue
        r = train_run(SHAPE, TOKENS, train, val, unique=TOKENS / ep)
        print(f"epochs {r['epochs']:5.1f}  unique {r['unique_tokens'] / 1e6:6.2f}M  val {r['val_loss']:.4f}  "
              f"{r['train_seconds']:.0f}s", flush=True)
        with OUT.open("a") as f:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
