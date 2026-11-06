"""The mixture problem and Figure 06: find w* on a three-domain simplex from small runs.

Domains: prose (WikiText-2), code (Python standard library), math (synthetic arithmetic).
Target: the mean of the three per-domain validation losses, in bits per byte.

1. Train one small model per random mixture w (Dirichlet(1) draws). Record per-domain validation
   loss and a 4-way multiple-choice arithmetic test at the last few checkpoints.
2. --fit: fit a RegMix-style quadratic surrogate and a data mixing law to those runs, report
   leave-one-out error, pick each method's optimum on a simplex grid, and train at that optimum,
   at the uniform mixture and at a "hand-written table", to see which wins.

Run with --size s --fit for the search, and again with --size m to get the larger-model ranking
that 08_signal_noise.py uses as its target.
"""

import argparse

import _common
import numpy as np
from ptlab import corpora, mixing, tinylm

DOMAINS = ["prose", "code", "math"]


def make_eval(device, data, n_windows):
    val = {d: tinylm.eval_windows(data[d]["val"], n_windows, 256) for d in DOMAINS}
    prompts, options, answers = tinylm.math_choice_items(200)

    def extra(model):
        s = tinylm.choice_logprob(model, prompts, options, device)
        return {"mc_correct": (s.argmax(1) == np.array(answers)).astype(int).tolist()}

    return val, extra


def summarize(run: dict) -> dict:
    """Keep small numbers only: per checkpoint, mean bpb on all windows and on the first 8."""
    cks = []
    for c in run["checkpoints"]:
        row = {"step": c["step"]}
        for d in DOMAINS:
            row[d] = tinylm.mean_bpb(c, d)
            row[d + "_8"] = tinylm.mean_bpb(c, d, 8)
        if "mc_correct" in c:
            row["mc_acc"] = float(np.mean(c["mc_correct"]))
            row["mc_acc_20"] = float(np.mean(c["mc_correct"][:20]))
        cks.append(row)
    return {"weights": run["config"]["weights"], "params": run["params"], "seconds": run["seconds"], "checkpoints": cks}


def target(row: dict) -> float:
    return float(np.mean([row[d] for d in DOMAINS]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--size", default="s")
    ap.add_argument("--runs", type=int, default=16)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--eval-windows", type=int, default=256)
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    device = _common.pick_device(args.device)
    name = args.name or f"07_mixture_{args.size}"
    out = _common.Report(name, device)
    data = corpora.load_domains()
    train = {d: data[d]["train"] for d in DOMAINS}
    val, extra = make_eval(device, data, args.eval_windows)
    out(f"train bytes: " + ", ".join(f"{d} {len(train[d]):,}" for d in DOMAINS))

    def run(w, seed=0):
        cfg = tinylm.RunConfig(weights=dict(zip(DOMAINS, map(float, w))), size=args.size, steps=args.steps,
                               eval_every=args.steps, last_k=5, last_every=max(args.steps // 30, 1), seed=seed)
        r = tinylm.train_run(cfg, train, val, device, extra_eval=extra)
        r.pop("model")
        return summarize(r)

    W = mixing.sample_simplex(args.runs, 3, seed=0)
    runs = []
    out(f"\n{len(W)} random mixtures, model size {args.size}, {args.steps} steps of 32 x 256 bytes")
    out(f"{'run':>3} {'w prose':>8} {'w code':>7} {'w math':>7} {'prose':>7} {'code':>7} {'math':>7} {'target':>7} "
        f"{'MC acc':>7}")
    for i, w in enumerate(W):
        r = run(w)
        runs.append(r)
        f = r["checkpoints"][-1]
        out(f"{i:>3} {w[0]:8.3f} {w[1]:7.3f} {w[2]:7.3f} {f['prose']:7.3f} {f['code']:7.3f} {f['math']:7.3f} "
            f"{target(f):7.3f} {f['mc_acc']:7.3f}")
    out(f"params {runs[0]['params']:,}; mean run time {np.mean([r['seconds'] for r in runs]):.0f} s. "
        "Losses are bits per byte at the final step.")
    result = {"size": args.size, "steps": args.steps, "runs": runs}

    if args.fit:
        y = np.array([target(r["checkpoints"][-1]) for r in runs])
        L = np.array([[r["checkpoints"][-1][d] for d in DOMAINS] for r in runs])
        grid = mixing.simplex_grid(3, 0.02)
        reg = mixing.QuadraticSurrogate().fit(W, y)
        law = mixing.MixingLaw().fit(W, L)
        w_reg = grid[np.argmin(reg.predict(grid))]
        w_law = grid[np.argmin(law.predict(grid).mean(1))]
        mae_reg = mixing.leave_one_out_mae(mixing.QuadraticSurrogate, W, y)

        class LawTarget:
            def fit(self, w, yy):
                self.m = mixing.MixingLaw().fit(w, L[[np.where((W == x).all(1))[0][0] for x in w]])
                return self

            def predict(self, w):
                return self.m.predict(w).mean(1)

        mae_law = mixing.leave_one_out_mae(LawTarget, W, y)
        out(f"\nLeave-one-out error on the target (bpb): quadratic surrogate {mae_reg:.4f}, mixing law {mae_law:.4f}")
        out(f"spread of the target across the {len(W)} runs: {y.min():.3f} to {y.max():.3f}")
        cands = {"uniform 1/3 each": np.ones(3) / 3, "hand table 60/25/15": np.array([0.6, 0.25, 0.15]),
                 "best random run": W[int(np.argmin(y))], "RegMix-style optimum": w_reg, "mixing-law optimum": w_law}
        out(f"\n{'mixture':<22} {'w prose':>8} {'w code':>7} {'w math':>7} {'predicted':>10} {'measured':>9} "
            f"{'seed 1':>7}")
        checks = {}
        for label, w in cands.items():
            pred = float(reg.predict(w[None])[0]) if "RegMix" in label else float(law.predict(w[None]).mean())
            r0, r1 = run(w, 0), run(w, 1)
            m0, m1 = target(r0["checkpoints"][-1]), target(r1["checkpoints"][-1])
            checks[label] = {"w": w.tolist(), "pred": pred, "seed0": m0, "seed1": m1}
            out(f"{label:<22} {w[0]:8.2f} {w[1]:7.2f} {w[2]:7.2f} {pred:10.3f} {m0:9.3f} {m1:7.3f}")
        out("'predicted' is the mixing law's prediction except on the RegMix row. Seeds change init and sampling.")
        out("Near the optimum the surface is flat: compare the gaps between rows with the seed-to-seed gap.")
        result["fit"] = {"loo_mae_quadratic": mae_reg, "loo_mae_law": mae_law, "checks": checks}
    _common.save_json(name, result, device)
    out.save()


if __name__ == "__main__":
    main()
