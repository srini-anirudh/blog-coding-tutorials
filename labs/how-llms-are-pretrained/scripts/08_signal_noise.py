"""Figure 11 and "Knowing whether it worked": signal, noise and decision accuracy on real small runs.

Reads the runs written by 07_mixture_search.py at two model sizes (the same random mixtures). Each
mixture is a "data recipe". For every metric measured on the small models we compute
- signal: spread of the final score across recipes,
- noise: spread of one run's score over its last 5 checkpoints, after removing a straight-line trend
  (the loss is still falling on the cosine schedule, and a trend is not noise),
- decision accuracy: share of recipe pairs ordered the same way as by the larger model.
Metrics are per-domain loss on 256 windows or on only 8 windows, and the 4-way arithmetic test
on 200 items or on 20. Small eval sets play the role of small, noisy benchmarks.
"""

import argparse
import json

import _common
import numpy as np
from ptlab import evalnoise

DOMAINS = ["prose", "code", "math"]


def metric(ck: dict, key: str) -> float:
    if key == "mean":
        return float(np.mean([ck[d] for d in DOMAINS]))
    if key == "mean_8":
        return float(np.mean([ck[d + "_8"] for d in DOMAINS]))
    if key.startswith("mc_acc"):
        return 1 - ck[key]  # error rate, so lower is better like the losses
    return ck[key]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", default="07_mixture_s")
    ap.add_argument("--large", default="07_mixture_m")
    args = ap.parse_args()
    small = json.loads((_common.RESULTS / f"{args.small}.json").read_text())
    large = json.loads((_common.RESULTS / f"{args.large}.json").read_text())
    out = _common.Report("08_signal_noise")
    out(f"computed from results/{args.small}.json ({small['hardware']})")
    out(f"and results/{args.large}.json ({large['hardware']})")
    rs, rl = small["runs"], large["runs"]
    n = min(len(rs), len(rl))
    rs, rl = rs[:n], rl[:n]
    out(f"{n} recipes. Small model {rs[0]['params']:,} params, large model {rl[0]['params']:,} params.\n")
    keys = [("prose", "prose"), ("code", "code"), ("math", "math"), ("mean", "mean"),
            ("prose_8", "prose"), ("code_8", "code"), ("math_8", "math"), ("mean_8", "mean"),
            ("mc_acc", "mc_acc"), ("mc_acc_20", "mc_acc")]
    out(f"{'small-model metric':<26} {'signal':>8} {'noise':>8} {'SNR':>7} {'decision acc.':>14}")
    rows = []
    for k, truth_key in keys:
        final = np.array([metric(r["checkpoints"][-1], k) for r in rs])
        last = np.array([[metric(c, k) for c in r["checkpoints"][-5:]] for r in rs])
        truth = np.array([metric(r["checkpoints"][-1], truth_key) for r in rl])
        sig, noi = evalnoise.signal(final), evalnoise.noise(last, detrend=True)
        da = evalnoise.decision_accuracy(final, truth)
        rows.append(dict(metric=k, signal=sig, noise=noi, snr=sig / max(noi, 1e-12), decision_accuracy=da))
        label = {"mc_acc": "arith. test error, 200 items", "mc_acc_20": "arith. test error, 20 items"}.get(
            k, (k.replace("_8", "") + " bpb, " + ("8 windows" if k.endswith("_8") else "256 windows")))
        out(f"{label:<26} {sig:8.4f} {noi:8.4f} {sig / max(noi, 1e-12):7.1f} {da:14.2f}")
    snr = np.log([r["snr"] for r in rows])
    da = np.array([r["decision_accuracy"] for r in rows])
    r2 = float(np.corrcoef(snr, da)[0, 1] ** 2)
    out(f"\nR^2 between log SNR and decision accuracy over these {len(rows)} metrics: {r2:.2f}")
    out("Ten metrics from one toy suite is a tiny sample; read it as a direction, not a measurement.")
    acc = [1 - metric(r["checkpoints"][-1], "mc_acc") for r in rs]
    acc_l = [1 - metric(r["checkpoints"][-1], "mc_acc") for r in rl]
    out(f"\nArithmetic test accuracy, chance = 0.25: small models {min(acc):.2f} to {max(acc):.2f}, "
        f"large models {min(acc_l):.2f} to {max(acc_l):.2f}")

    t = np.array([metric(r["checkpoints"][-1], "mean") for r in rs])
    i, j = sorted(range(n), key=lambda x: t[x])[:2]
    out(f"\nThe two best recipes on the small model (runs {i} and {j}), last 5 checkpoints:")
    for k in ["mean", "mean_8"]:
        a = np.array([metric(c, k) for c in rs[i]["checkpoints"][-5:]])
        b = np.array([metric(c, k) for c in rs[j]["checkpoints"][-5:]])
        out(f"  {k:<7} run {i}: {np.round(a, 4).tolist()}  run {j}: {np.round(b, 4).tolist()}  "
            f"gap {abs(a.mean() - b.mean()):.4f}, checkpoints where the order flips: {evalnoise.rank_flip_rate(a, b):.0%}")
    _common.save_json("08_signal_noise", {"rows": rows, "r2_logsnr_decision": r2})
    out.save()


if __name__ == "__main__":
    main()
