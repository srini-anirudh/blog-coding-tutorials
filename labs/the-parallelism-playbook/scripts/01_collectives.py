"""The collectives: a ring AllReduce built from send/recv, the identity AllReduce = ReduceScatter + AllGather,
and (with --bench) the alpha-beta cost model T = alpha + n / beta fitted to measured times."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from ppbook import workers  # noqa: E402
from ppbook.dist_utils import Report, spawn  # noqa: E402


def fit_alpha_beta(points):
    """t = alpha + n / beta. alpha is the fastest of the three smallest messages (latency regime). 1 / beta is the
    least-squares slope over the messages of at least 1/16 of the largest (bandwidth regime). One fit over sizes
    spanning five orders of magnitude would let the large messages decide alpha. Returns (us, GB/s)."""
    pts = sorted(points)
    alpha = min(t for _, t in pts[:3])
    big = [(n, t) for n, t in pts if n >= pts[-1][0] / 16]
    nb, tb = np.array([p[0] for p in big], float), np.array([p[1] for p in big], float)
    slope = np.polyfit(nb, tb, 1)[0]
    return alpha * 1e6, 1 / max(slope, 1e-15) / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true", help="time the collectives over a range of message sizes")
    ap.add_argument("--world", type=int, default=2)
    ap.add_argument("--max-mb", type=float, default=64)
    ap.add_argument("--out", default="01_collectives.txt")
    args = ap.parse_args()
    rep = Report(args.out)
    if not args.bench:
        rep("ring AllReduce from send/recv, and ReduceScatter followed by AllGather, against dist.all_reduce")
        rep(f"{'ranks':>5} {'ring vs native':>15} {'RS+AG vs AR':>12} {'sent / buffer':>14} {'2(N-1)/N':>9}")
        for n in (2, 3, 4, 8):
            r = spawn(workers.collectives_check, n, 1024 * n)[0]
            rep(f"{n:>5} {r['ring_vs_native']:>15.2e} {r['rs_then_ag_vs_allreduce']:>12.2e} {r['ratio']:>14.3f} "
                f"{2 * (n - 1) / n:>9.3f}")
        rep("")
        rep("Each rank sends 2(N-1)/N of the buffer whatever N is: per-rank traffic converges to 2x the buffer.")
    else:
        sizes = [int(2 ** k) for k in range(12, 40) if 2 ** k <= args.max_mb * 2 ** 20]
        res = spawn(workers.bench_collectives, args.world, sizes)[0]
        rep(f"{args.world} ranks; time per call = 20 back-to-back calls / 20; send_recv is rank 0 -> rank 1")
        rep(f"{'bytes':>12} " + " ".join(f"{k:>15}" for k in res))
        for i in range(len(sizes)):
            rep(f"{res['all_reduce'][i][0]:>12} " + " ".join(f"{res[k][i][1] * 1e6:>13.1f}us" for k in res))
        rep("")
        rep("alpha-beta fit, t = alpha + n / beta (n = bytes in each rank's buffer; alpha from the smallest messages, beta fitted over the largest)")
        for k, pts in res.items():
            a, b = fit_alpha_beta(pts)
            big = pts[-1]
            rep(f"  {k:>15}: alpha {a:8.1f} us, beta {b:8.2f} GB/s; largest message {big[0] / big[1] / 1e9:.2f} GB/s")
        n = args.world
        ar = res["all_reduce"][-1]
        rep(f"  all_reduce bus bandwidth at {ar[0] >> 20} MB: {2 * (n - 1) / n * ar[0] / ar[1] / 1e9:.2f} GB/s "
            "(bytes each rank actually sends per second)")
        a_ar, b_ar = fit_alpha_beta(res["all_reduce"])
        rep(f"  all_reduce below about {a_ar * 1e-6 * b_ar * 1e9 / 2**20:.2f} MB (alpha x beta) costs mostly latency, not")
        rep("  bandwidth, which is why DDP fuses gradients into buckets")
    rep.save()


if __name__ == "__main__":
    main()
