"""A slow all-reduce is usually a fast rank waiting (Figure 06).

Four processes on one machine (gloo backend, CPU tensors). Each rank "computes" by sleeping 40 ms, then
all-reduces a buffer. In the skewed scenario one rank (rank 2, or the last rank if there are fewer) computes 18 ms longer. We time the all-reduce on every
rank, then split it with a barrier placed just before it: time in the barrier is arrival skew (load
imbalance), time after it is transfer. Last, we halve the message size to tell bandwidth from latency.

The mechanism is the same on NCCL and GPUs; the absolute times here are CPU and shared memory, not a network.

    python scripts/05_distributed_skew.py
    python scripts/05_distributed_skew.py --backend nccl --world 2   # one GPU per rank
"""

import argparse
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from whyslow.bench import Report  # noqa: E402

COMPUTE_MS, SKEW_MS, ITERS = 40.0, 18.0, 12


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def timed(fn, dev):
    sync(dev)
    t = time.perf_counter()
    fn()
    sync(dev)
    return (time.perf_counter() - t) * 1e3


def worker(rank, world, init, backend, out_q):
    slow = min(2, world - 1)
    torch.set_num_threads(1)
    dev = torch.device("cuda", rank) if backend == "nccl" else torch.device("cpu")
    dist.init_process_group(backend, init_method=init, rank=rank, world_size=world)
    buf = torch.ones(4 << 20, device=dev)  # 16 MiB fp32
    res = {}
    for scen, skew in [("balanced", 0.0), ("one rank slow", SKEW_MS)]:
        ar, bar, xfer, comp = [], [], [], []
        for i in range(ITERS + 2):
            t0 = time.perf_counter()
            time.sleep((COMPUTE_MS + (skew if rank == slow else 0.0)) / 1e3)  # the forward + backward
            c = (time.perf_counter() - t0) * 1e3
            a = timed(lambda: dist.all_reduce(buf), dev)
            time.sleep((COMPUTE_MS + (skew if rank == slow else 0.0)) / 1e3)
            b = timed(lambda: dist.barrier(), dev)
            x = timed(lambda: dist.all_reduce(buf), dev)
            if i >= 2:
                ar.append(a), bar.append(b), xfer.append(x), comp.append(c)
        med = statistics.median
        res[scen] = {"compute_ms": med(comp), "allreduce_ms": med(ar), "barrier_ms": med(bar), "transfer_ms": med(xfer)}
    sizes = {}
    for numel in (16 << 20, 8 << 20, 4 << 20, 1024, 512):
        t = torch.ones(numel, device=dev)
        dist.barrier()
        ts = []
        for i in range(ITERS + 2):
            dist.barrier()
            ts.append(timed(lambda: dist.all_reduce(t), dev))
        sizes[numel * 4] = statistics.median(ts[2:])
    res["sizes"] = sizes
    out_q.put((rank, res))
    dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="gloo", choices=["gloo", "nccl"])
    ap.add_argument("--world", type=int, default=4)
    args = ap.parse_args()
    r = Report("05_distributed_skew" + ("" if args.backend == "gloo" else "_nccl"))
    with tempfile.TemporaryDirectory() as d:
        init = "file://" + os.path.join(d, "store")
        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        procs = [ctx.Process(target=worker, args=(rk, args.world, init, args.backend, q)) for rk in range(args.world)]
        for p in procs:
            p.start()
        results = dict(q.get() for _ in procs)
        for p in procs:
            p.join()
    r(f"{args.world} ranks, backend {args.backend}, compute {COMPUTE_MS:g} ms per step, 16 MiB all-reduce, "
      f"median of {ITERS} steps.\n")
    r(f"{'scenario':13s} {'rank':>4s} {'compute':>8s} {'all-reduce':>10s} | {'barrier':>8s} {'transfer':>8s}  (ms)")
    for scen in ("balanced", "one rank slow"):
        for rk in range(args.world):
            s = results[rk][scen]
            r(f"{scen:13s} {rk:4d} {s['compute_ms']:8.1f} {s['allreduce_ms']:10.1f} | {s['barrier_ms']:8.1f} "
              f"{s['transfer_ms']:8.1f}")
        comp = [results[rk][scen]["compute_ms"] for rk in range(args.world)]
        r(f"{'':13s} per-rank compute spread (max - min): {max(comp) - min(comp):.1f} ms\n")
    r("Halve the message. Time roughly halves: bandwidth bound. Time stays put: latency bound. (rank 0)")
    sz = results[0]["sizes"]
    keys = sorted(sz, reverse=True)
    for big, small in zip(keys, keys[1:]):
        r(f"  {big:>10d} B -> {small:>10d} B : {sz[big]:7.3f} -> {sz[small]:7.3f} ms  (ratio {sz[small] / sz[big]:.2f})")
    r.data = {"results": {str(k): v for k, v in results.items()}}
    return r.save()


if __name__ == "__main__":
    main()
