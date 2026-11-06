"""Timing tensor parallelism. (1) One block's forward + backward on 1 rank and on T ranks, with real
collectives and with the wire skipped: the difference is communication the step could not hide.
(2) GEMM throughput as TP narrows the per-rank matmul. Meant for GPUs; on CPU use small sizes."""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from ppbook import workers  # noqa: E402
from ppbook.dist_utils import Report, spawn  # noqa: E402


def gemm_tflops(m, k, n, dtype, dev, iters=20):
    a, b = torch.randn(m, k, device=dev, dtype=dtype), torch.randn(k, n, device=dev, dtype=dtype)
    for _ in range(3):
        a @ b
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        a @ b
    if dev.type == "cuda":
        torch.cuda.synchronize()
    return 2 * m * k * n * iters / (time.perf_counter() - t0) / 1e12


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", type=int, default=2)
    ap.add_argument("--configs", default="1024:16:2048:4,4096:32:2048:4,8192:64:2048:2",
                    help="hidden:heads:seq:batch, comma separated")
    ap.add_argument("--out", default="08_bench_tp.txt")
    args = ap.parse_args()
    cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if cuda else torch.float32
    rep = Report(args.out)
    rep(f"(1) one block, forward + backward, {dtype}; time per step from 10 back-to-back steps, median of 3 such loops. fake = collectives replaced by local")
    rep("no-ops (same shapes, wrong values): it times the compute alone. exposed = real - fake. held = memory the")
    rep("forward pass leaves allocated for backward (activations, plus the output); peak fwd includes transients.")
    rep(f"{'h':>5} {'s':>5} {'b':>2} {'variant':>16} {'step ms':>8} {'fake ms':>8} {'exposed ms':>11} {'exposed %':>10}"
        f" {'peak fwd MB':>12} {'held MB':>8} {'AR msg MB':>10}")
    for cfg in args.configs.split(","):
        h, heads, s, b = map(int, cfg.split(":"))
        msg = s * b * h * (2 if cuda else 4) / 2**20
        for name, world, sp, rg in (("1 rank", 1, False, False), (f"TP {args.world}", args.world, False, False),
                                    (f"TP {args.world}+SP", args.world, True, False),
                                    (f"TP {args.world}+SP regat.", args.world, True, True)):
            r = spawn(workers.bench_tp, world, h, heads, s, b, sp, rg, 10, dtype)[0]
            rep(f"{h:>5} {s:>5} {b:>2} {name:>16} {r['real'] * 1e3:>8.2f} {r['fake'] * 1e3:>8.2f} "
                f"{r['exposed_comm_s'] * 1e3:>11.2f} {r['exposed_comm_s'] / r['real']:>10.1%} "
                f"{r.get('peak_fwd_mem_bytes', 0) / 2**20:>12.0f} {r.get('held_act_bytes', 0) / 2**20:>8.0f} {msg if world > 1 else 0:>10.1f}")
    rep("")
    dev = torch.device("cuda" if cuda else "cpu")
    rep(f"(2) GEMM [b*s, h] x [h, 4h/T] on one device, b*s = 8192, {dtype}: what each TP rank's MLP up-projection costs")
    rep(f"{'h':>5} " + " ".join(f"{'T=' + str(t) + ' TFLOP/s':>14}" for t in (1, 2, 4, 8)) + "   T=8 / T=1")
    for h in (1024, 4096, 8192):
        tf = [gemm_tflops(8192, h, 4 * h // t, dtype, dev) for t in (1, 2, 4, 8)]
        rep(f"{h:>5} " + " ".join(f"{x:>14.1f}" for x in tf) + f"   {tf[-1] / tf[0]:.2f}")
    rep.save()


if __name__ == "__main__":
    main()
