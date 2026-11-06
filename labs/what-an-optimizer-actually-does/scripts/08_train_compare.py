"""Seven optimizers train the same small language model for the same number of steps.

TinyLLM (C4), 0.77 M parameters, byte-level Tiny Shakespeare, 300 steps of 16 x 128 tokens,
warmup then cosine decay, gradient clipping at 1.0. Each optimizer gets a 3 to 5 point lr sweep and the
best final validation loss is reported (a sweep that peaks at its edge was widened), with its state bytes per parameter and the
update-to-weight RMS ratio of every 2-D tensor on the last step.

This is a small-scale look at the mechanisms. It does not reproduce any paper's large-scale ranking.

Run everything (about 20 min on 8 CPU threads):
  python labs/what-an-optimizer-actually-does/scripts/08_train_compare.py
Or one optimizer at a time, then merge:
  python labs/what-an-optimizer-actually-does/scripts/08_train_compare.py --opt AdamW
  python labs/what-an-optimizer-actually-does/scripts/08_train_compare.py --table
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from optlab import SGD, Adafactor, Adam, AdamW8bit, Hybrid, Lion, Muon
from optlab.common import RESULTS, Report, hardware, split_for_muon
from optlab.train import train

WD = 0.1
ADAM_LRS = [1e-3, 3e-3, 1e-2]


def muon_hybrid(m, lr):
    hidden, rest = split_for_muon(m)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    return Hybrid(Muon(hidden, lr=lr, weight_decay=WD, ns_dtype=dtype), Adam(rest, lr=lr, weight_decay=WD))


OPTS = {
    "SGD + momentum": ([0.03, 0.1, 0.3], lambda m, lr: SGD(m.parameters(), lr, momentum=0.9)),
    "AdamW": (ADAM_LRS, lambda m, lr: Adam(m.parameters(), lr, weight_decay=WD)),
    "Adam + L2": (ADAM_LRS, lambda m, lr: Adam(m.parameters(), lr, weight_decay=WD, decoupled=False)),
    # The same wd number means a much stronger pull when it goes through the adaptive denominator.
    "Adam + L2 1e-3": (ADAM_LRS, lambda m, lr: Adam(m.parameters(), lr, weight_decay=1e-3, decoupled=False)),
    "8-bit AdamW": (ADAM_LRS, lambda m, lr: AdamW8bit(m.parameters(), lr, weight_decay=WD)),
    # Lion: lr about 10x lower than AdamW, wd 10x higher, so lr * wd stays the same.
    "Lion": ([1e-4, 3e-4, 1e-3, 3e-3], lambda m, lr: Lion(m.parameters(), lr, weight_decay=10 * WD)),
    "Adafactor": (ADAM_LRS, lambda m, lr: Adafactor(m.parameters(), lr, weight_decay=WD)),
    "Muon + AdamW": ([1e-3, 3e-3, 1e-2, 3e-2, 1e-1], muon_hybrid),
}
PART = RESULTS / "08_train_parts"


def run_one(name):
    lrs, make = OPTS[name]
    runs = []
    for lr in lrs:
        r = train(lambda m: make(m, lr))
        r.pop("model", None), r.pop("opt", None)
        r["lr"] = lr
        runs.append(r)
        print(f"{name} lr={lr}: " + ("diverged" if r["diverged"] else f"val {r['val_loss']:.4f}"), flush=True)
    PART.mkdir(parents=True, exist_ok=True)
    out = {"hardware": hardware(), "name": name, "runs": runs}
    (PART / f"{name.replace(' ', '_').replace('+', 'plus')}.json").write_text(json.dumps(out, indent=1) + "\n")


def seed_check():
    """Repeat the best setting of three optimizers with seeds 1 and 2 (seed 0 is in the sweep)."""
    rep = Report("08_seed_check")
    rep("best lr from the sweep, new seeds change the model init and the data order; val loss per seed")
    best = {"AdamW": 3e-3, "8-bit AdamW": 3e-3, "Muon + AdamW": 1e-2}
    out = {}
    for name, lr in best.items():
        vals = []
        for seed in (1, 2):
            r = train(lambda m: OPTS[name][1](m, lr), seed=seed)
            vals.append(r["val_loss"])
        out[name] = vals
        rep(f"{name:<14} lr {lr:g}: seed 1 {vals[0]:.3f}  seed 2 {vals[1]:.3f}")
    rep.save({"val_loss_by_seed": out})


def table():
    rep = Report("08_train_compare")
    rep("TinyLLM 0.77 M params, Tiny Shakespeare bytes, 300 steps x 16 x 128 tokens, warmup 30, cosine to 10%")
    rep("val loss = mean over 10 held-out batches; ratio = RMS(dW)/RMS(W) per 2-D tensor on the last step")
    rep(f"{'optimizer':<15} {'lr sweep: val loss':<30} {'best':>7} {'B/param':>8} {'s/run':>6} "
        f"{'ratio median [min, max]':>26}")
    summary = {}
    for name in OPTS:
        f = PART / f"{name.replace(' ', '_').replace('+', 'plus')}.json"
        if not f.exists():
            rep(f"{name:<15} (not run)")
            continue
        part = json.loads(f.read_text())
        runs = part["runs"]
        sweep = " ".join("div" if r["diverged"] else f"{r['val_loss']:.3f}" for r in runs)
        ok = [r for r in runs if not r["diverged"]]
        best = min(ok, key=lambda r: r["val_loss"])
        ratios = list(best["update_ratio"].values())
        rep(f"{name:<15} {sweep:<30} {best['val_loss']:>7.3f} {best['state_bytes_per_param']:>8.3f} "
            f"{best['seconds']:>6.0f} {statistics.median(ratios):>9.2e} [{min(ratios):.1e}, {max(ratios):.1e}]")
        summary[name] = {"best_lr": best["lr"], "val_loss": best["val_loss"], "train_loss": best["train_loss"],
                         "state_bytes_per_param": best["state_bytes_per_param"], "sweep": sweep,
                         "update_ratio_median": statistics.median(ratios), "hardware": part["hardware"]}
    rep("lr sweeps: SGD 0.03/0.1/0.3; Lion 1e-4 to 3e-3; Muon 1e-3 to 1e-1; all others 1e-3/3e-3/1e-2")
    rep.save({"results": summary})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--opt", choices=list(OPTS))
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--seed-check", action="store_true")
    a = ap.parse_args()
    if a.seed_check:
        seed_check()
    elif a.opt:
        run_one(a.opt)
    elif a.table:
        table()
    else:
        for name in OPTS:
            run_one(name)
        table()
