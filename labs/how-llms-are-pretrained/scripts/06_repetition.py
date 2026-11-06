"""Figure 10 and "What is an epoch?": the value of repeated data.

Part 1 evaluates the effective-data formula of Muennighoff et al. (2023) with their fitted R* = 15.4.
Part 2 is a small experiment. Every run sees the same number of training bytes, but drawn from a
pool of U = budget / E unique bytes of WikiText-2, so the pool is repeated about E times. A
single-epoch run (E = 1) gives the fresh-data curve: validation loss against bytes seen. For each
E we ask how many fresh bytes would give the same final loss. That is the measured effective data.
The paper fitted runs up to 9B parameters; this is a 0.6M-parameter byte model, so expect the shape, not the numbers.
"""

import argparse

import _common
import numpy as np
from ptlab import corpora, repetition, tinylm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--size", default="s")
    ap.add_argument("--epochs", default="1,2,4,8,16,32,64")
    ap.add_argument("--eval-windows", type=int, default=256)
    ap.add_argument("--name", default="06_repetition")
    args = ap.parse_args()
    device = _common.pick_device(args.device)
    out = _common.Report(args.name, device)

    out(f"Part 1. D'/(U E): value of the data actually seen, relative to fresh data, R* = {repetition.R_STAR_PAPER}")
    es = [1, 2, 4, 8, 16, 40, 100]
    out("  epochs E   " + "".join(f"{e:>7}" for e in es))
    out("  D'/(U E)   " + "".join(f"{v:7.3f}" for v in repetition.value_fraction(es)))
    out("  D'/U       " + "".join(f"{v:7.2f}" for v in repetition.effective_tokens(1.0, es)))
    out(f"  ceiling as E grows: D'/U -> 1 + R* = {1 + repetition.R_STAR_PAPER:.1f}\n")

    train = corpora.as_bytes(corpora.wikitext_docs("train"))
    val = {"prose": tinylm.eval_windows(corpora.as_bytes(corpora.wikitext_docs("validation")), args.eval_windows, 256)}
    batch, seq = 32, 256
    budget = args.steps * batch * seq
    assert budget <= len(train), "budget larger than the WikiText-2 train split"
    runs = {}
    for e in [int(x) for x in args.epochs.split(",")]:
        u = budget // e
        cfg = tinylm.RunConfig(weights={"prose": 1.0}, size=args.size, steps=args.steps, batch=batch, seq_len=seq,
                               lr=2e-3, schedule="constant", eval_every=max(args.steps // 40, 1),
                               unique_bytes={"prose": u}, seed=0)
        r = tinylm.train_run(cfg, {"prose": train}, val, device)
        r.pop("model")
        runs[e] = r
        out(f"  trained E={e:<3} U={u:>10,} bytes  final val {tinylm.mean_bpb(r['checkpoints'][-1], 'prose'):.4f} bpb "
            f"({r['seconds']:.0f} s, {r['params']:,} params)")
    fresh = runs[1]["checkpoints"]
    fb = np.array([c["bytes_seen"] for c in fresh], dtype=float)
    fl = np.array([tinylm.mean_bpb(c, "prose") for c in fresh])

    out(f"\nPart 2. Same budget of {budget:,} training bytes for every run, constant learning rate after warmup.")
    out(f"{'E':>4} {'unique U':>11} {'val bpb':>8} {'best bpb':>9} {'measured D_eff':>15} {'D_eff/U':>8} "
        f"{'formula D/U':>12} {'value kept':>11}")
    us, eps, meas = [], [], []
    rows = []
    for e, r in runs.items():
        u = budget // e
        final = tinylm.mean_bpb(r["checkpoints"][-1], "prose")
        best = min(tinylm.mean_bpb(c, "prose") for c in r["checkpoints"])
        d_eff = repetition.interp_fresh(fb, fl, final)
        mark = " "
        if not np.isfinite(d_eff):  # lower loss than the fresh run ever reached: count it as fresh
            d_eff, mark = float(budget), "*"
        us.append(u), eps.append(e), meas.append(d_eff)
        rows.append(dict(epochs=e, unique=u, final_bpb=final, best_bpb=best, d_eff=d_eff))
        out(f"{e:>4} {u:>11,} {final:8.4f} {best:9.4f} {d_eff:>15,.0f} {d_eff / u:8.2f} "
            f"{float(repetition.effective_tokens(1.0, e)):12.2f} {d_eff / budget:11.2f}{mark}")
    r_fit = repetition.fit_r_star(np.array(us, float), np.array(eps, float), np.array(meas))
    out(f"\nR* fitted to these runs: {r_fit:.2f} (paper: {repetition.R_STAR_PAPER}, fitted at far larger scale)")
    if any(not np.isfinite(r["d_eff"]) or r["d_eff"] == budget and r["epochs"] > 1 for r in rows):
        out("* a run that ended lower than the fresh run has its D_eff set to the full budget.")
    out("'value kept' = D_eff / bytes seen: 1.00 means repeats were worth as much as fresh bytes.")
    out("'best bpb' below 'val bpb' means the run overfit: validation loss rose again before the end.")
    _common.save_json(args.name, {"budget_bytes": budget, "rows": rows, "r_star_fit": r_fit,
                                  "fresh_curve": {"bytes": fb.tolist(), "bpb": fl.tolist()}}, device)
    out.save()


if __name__ == "__main__":
    main()
