"""Question 2 at small scale: do the connectors differ on "dense text" and "photograph" questions?

Same frozen patch encoder, same pretrained tiny LM, same data and steps. Only the connector
changes. The LM and connector train together ("unfreeze" schedule); the encoder stays frozen.
Each glyph page has 4 x 4 = 16 patches.

    python labs/the-vlm-architecture-gallery/scripts/05_connector_bench.py --seed 0
    python labs/the-vlm-architecture-gallery/scripts/05_connector_bench.py --report   # merge seeds

This is a toy: a 0.2M-parameter LM and synthetic glyphs. It shows the mechanism the post describes
(what each connector can and cannot keep). It does not reproduce any paper's benchmark numbers.
"""

import argparse
import json
import statistics

import _common
import torch

from gallery.model import TinyVLM
from gallery.train import evaluate, fresh_copy, pretrain_lm, train_vlm

CONNECTORS = [
    ("linear", {}, "1:1"),
    ("mlp", {}, "1:1"),
    ("pixel_shuffle", {"s": 2}, "fold 2x2"),
    ("avg_pool", {"k": 2}, "mean 2x2"),
    ("attn_pool", {"k": 2}, "attend 2x2"),
    ("resampler", {"n_queries": 4}, "4 queries"),
    ("resampler", {"n_queries": 16}, "16 queries"),
]


def run(seed: int, steps: int):
    _common.threads()
    lm = pretrain_lm(seed=seed, steps=1500, cache=_common.RUNS / f"lm_seed{seed}.pt")
    rows = []
    for name, kw, how in CONNECTORS:
        torch.manual_seed(seed)
        vlm = TinyVLM(fresh_copy(lm), connector=name, **kw)
        info = train_vlm(vlm, ("fine", "coarse"), "all", steps=steps, seed=seed)
        row = {"connector": name, "kwargs": kw, "how": how, "tokens": vlm.connector.out_tokens(*vlm.grid),
               "fine": evaluate(vlm, "fine", True), "coarse": evaluate(vlm, "coarse", True), **info}
        print(json.dumps(row), flush=True)
        rows.append(row)
    out = _common.RESULTS / f"05_connector_bench_seed{seed}.json"
    out.write_text(json.dumps({"hardware": _common.hardware(), "command": _common.command(),
                               "steps": steps, "seed": seed, "rows": rows}, indent=1) + "\n")


def report():
    files = sorted(_common.RESULTS.glob("05_connector_bench_seed*.json"))
    runs = [json.loads(f.read_text()) for f in files]
    rep = _common.Report("05_connector_bench")
    rep(f"# merged from {len(runs)} seeds: " + ", ".join(f.name for f in files))
    rep(f"# each seed run on: {runs[0]['hardware']}; {runs[0]['steps']} steps, batch 128, encoder frozen, LM + connector train")
    rep("accuracy on 2048 fresh pages, mean over seeds (min to max). chance is 0.125 for both questions.")
    rep(f"{'connector':<26}{'tokens':>7}{'fine: cell r,c':>22}{'coarse: majority':>22}{'params':>10}{'train s':>9}")
    merged = []
    for i, (name, kw, how) in enumerate(CONNECTORS):
        rs = [r["rows"][i] for r in runs]
        f = [r["fine"] for r in rs]
        c = [r["coarse"] for r in rs]
        label = f"{name} ({how})"
        rep(f"{label:<26}{rs[0]['tokens']:>7}{statistics.mean(f):>9.3f} ({min(f):.3f}-{max(f):.3f})"
            f"{statistics.mean(c):>9.3f} ({min(c):.3f}-{max(c):.3f}){rs[0]['trainable_params']:>10,}"
            f"{statistics.mean(r['train_seconds'] for r in rs):>9.0f}")
        merged.append({"connector": label, "tokens": rs[0]["tokens"], "fine": f, "coarse": c,
                       "trainable_params": rs[0]["trainable_params"]})
    rep("trainable params = connector + LM; the encoder is frozen and not counted.")
    rep.save({"rows": merged})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    report() if a.report else run(a.seed, a.steps)
