"""Section 11, for real: run the same AllReduce schedules on p processes and time them.

    python scripts/04_allreduce_measured.py                    # 8 CPU processes over gloo
    python scripts/04_allreduce_measured.py --backend nccl -p 2  # 2 GPUs over NCCL

First it measures alpha and beta with a ping-pong through the same code path, then it times naive, tree, ring,
Rabenseifner and the library's own all_reduce at each size, and compares each with the alpha-beta prediction.
"""

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from commcost import alphabeta as ab  # noqa: E402
from commcost import dist_run, workers  # noqa: E402
from commcost.report import Report, human_bytes, table  # noqa: E402

MODEL = {"naive": ab.t_naive, "tree": ab.t_tree, "ring": ab.t_ring, "rabenseifner": ab.t_rhd}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="gloo", choices=["gloo", "nccl"])
    ap.add_argument("-p", type=int, default=8)
    ap.add_argument("--max-log2", type=int, default=24)
    ap.add_argument("--iters", type=int, default=15)
    args = ap.parse_args()
    gpu = args.backend == "nccl"
    rep = Report(f"04_allreduce_measured_{args.backend}_p{args.p}", gpu=gpu,
                 note=f"{args.p} processes" + (" on GPUs" if gpu else ", 1 thread each"))

    sizes = [1 << k for k in range(2, args.max_log2 + 1, 2)]
    pp = dist_run.launch(2, workers.pingpong, (sizes, 30, args.backend), backend=args.backend)
    alpha, beta = ab.fit_alpha_beta([r["bytes"] for r in pp], [r["one_way_s"] for r in pp])
    rep(f"Ping-pong fit (2 processes): alpha = {alpha * 1e6:.1f} us, 1/beta = {1 / beta / 1e9:.2f} GB/s")
    rep()

    sizes = [1 << k for k in range(10, args.max_log2 + 1, 2)]
    algos = ["naive", "tree", "ring", "rabenseifner", "library"]
    rows = dist_run.launch(args.p, workers.allreduce_bench, (sizes, algos, args.iters, args.backend),
                           backend=args.backend)
    assert all(r["correct"] for r in rows), "an algorithm produced a wrong sum"
    rep(f"All {len(rows)} runs gave the exact sum on every rank.")
    rep("Measured median ms (model ms in brackets). 'library' is torch.distributed.all_reduce.")
    by = {(r["bytes"], r["algo"]): r["seconds"] for r in rows}
    out, winners = [], []
    for n in sorted({r["bytes"] for r in rows}):
        cells = [human_bytes(n)]
        for a in algos:
            t = by[(n, a)]
            cells.append(f"{t * 1e3:.2f}" + (f" ({MODEL[a](n, args.p, alpha, beta) * 1e3:.2f})" if a in MODEL else ""))
        ours = {a: by[(n, a)] for a in MODEL}
        pred = {a: MODEL[a](n, args.p, alpha, beta) for a in MODEL}
        cells += [min(ours, key=ours.get), min(pred, key=pred.get)]
        winners.append({"bytes": n, "measured": min(ours, key=ours.get), "model": min(pred, key=pred.get)})
        out.append(cells)
    for line in table(out, ["n", *algos, "fastest", "model says"]):
        rep(line)
    agree = sum(w["measured"] == w["model"] for w in winners)
    rep()
    rep(f"The model picks the measured winner at {agree} of {len(winners)} sizes.")

    # Refit: what alpha and beta would each algorithm's own timings imply? T = rounds * alpha + bytes_factor * n * beta
    rep()
    rep("Per-algorithm refit of T = rounds * alpha' + bytes * beta' to the p-way timings above:")
    p, lg = args.p, math.ceil(math.log2(args.p))
    coef = {"naive": (2 * (p - 1), 2 * (p - 1)), "tree": (2 * lg, 2 * lg),
            "ring": (2 * (p - 1), 2 * (p - 1) / p), "rabenseifner": (2 * lg, 2 * (p - 1) / p)}
    refit = {}
    for a, (k_round, k_byte) in coef.items():
        ns = sorted(n for (n, b) in by if b == a)
        ts = np.array([by[(n, a)] for n in ns])
        m = np.stack([np.full(len(ns), k_round), k_byte * np.array(ns, float)], 1) / ts[:, None]
        (a2, b2), *_ = np.linalg.lstsq(m, np.ones(len(ns)), rcond=None)
        refit[a] = {"alpha_s": float(a2), "beta_s_per_byte": float(b2)}
        rep(f"  {a:13s} alpha' = {a2 * 1e6:7.1f} us per round, 1/beta' = {1 / b2 / 1e9:.2f} GB/s")
    rep(f"Against the 2-process ping-pong: alpha = {alpha * 1e6:.1f} us, 1/beta = {1 / beta / 1e9:.2f} GB/s.")
    rep.save({"alpha_s": alpha, "beta_s_per_byte": beta, "pingpong": pp, "rows": rows, "winners": winners,
              "refit": refit})


if __name__ == "__main__":
    main()
