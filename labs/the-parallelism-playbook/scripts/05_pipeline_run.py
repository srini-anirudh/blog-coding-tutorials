"""A real pipeline: one stage per process, activations and gradients over send/recv, GPipe or 1F1B order.
Without flags it checks gradients against one device on CPU. With --bench it times the schedules and reads
peak memory per stage (meaningful on GPUs; on CPU the timings are dominated by process overheads)."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from ppbook import workers  # noqa: E402
from ppbook.dist_utils import Report, spawn  # noqa: E402
from ppbook.pipeline import bubble_formula  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--stages", type=int, default=2)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=32, help="global batch, split into m microbatches")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if not args.bench:
        rep = Report(args.out or "05_pipeline_run.txt")
        rep("Stack(hidden 32, 4 heads, 4 blocks), 8 microbatches of 2 x seq 8, fp32, MSE loss averaged over microbatches")
        rep(f"{'P':>2} {'schedule':>9} {'max grad err vs 1 device':>25} {'loss (last stage)':>18} {'ref loss':>9}"
            f"  peak stashes per stage")
        for P in (2, 4):
            for sch in ("gpipe", "1f1b"):
                rs = spawn(workers.pipeline_run, P, sch, 8)
                err = max(r["grad_err"] for r in rs)
                rep(f"{P:>2} {sch:>9} {err:>25.1e} {rs[-1]['loss']:>18.6f} {rs[-1]['ref_loss']:>9.6f}  "
                    f"{[r['peak_stashes'] for r in rs]}")
        rep.save()
        return
    P = args.stages
    rep = Report(args.out or f"05_pipeline_bench_P{P}.txt")
    rep(f"Stack(hidden {args.hidden}, 16 heads, {args.layers} blocks) split over {P} stages, bf16, global batch "
        f"{args.batch} x seq {args.seq}; median of 5 steps after warm-up. busy = time inside forward/backward on")
    rep("the stage; measured bubble = 1 - busy / step time (it also absorbs transfer and launch gaps).")
    rep(f"{'sched':>6} {'m':>3} {'step ms':>8} {'stage busy ms':>16} {'bubble, worst stage':>20} {'formula':>8}"
        f" {'peak MB per stage':>20} {'peak stashes':>13}")
    for m in (1, 2, 4, 8, 16, 32):
        if args.batch % m:
            continue
        for sch in ("gpipe", "1f1b"):
            rs = spawn(workers.pipeline_run, P, sch, m, args.hidden, 16, args.layers, args.seq, args.batch // m,
                       0, 5, torch.bfloat16)
            mem = [r.get("peak_mem_bytes", 0) / 2**20 for r in rs]
            rep(f"{sch:>6} {m:>3} {rs[0]['step_s'] * 1e3:>8.1f} {str([round(r['busy_s'] * 1e3) for r in rs]):>16} "
                f"{max(r['measured_bubble'] for r in rs):>20.3f} {bubble_formula(P, m):>8.3f} "
                f"{str([round(x) for x in mem]):>20} {str([r['peak_stashes'] for r in rs]):>13}")
    rep.save()


if __name__ == "__main__":
    main()
