"""Functions that run inside each spawned process. They live in the package so `spawn` can import them."""

from __future__ import annotations

import time

import torch
import torch.distributed as dist

from . import schedules
from .dist_run import compile_for_rank, execute, sync, time_op


def _device(backend: str, rank: int) -> torch.device:
    return torch.device("cuda", rank) if backend == "nccl" else torch.device("cpu")


def pingpong(rank, world, sizes, iters, backend):
    """Rank 0 sends n bytes to rank 1 and waits for n bytes back. One-way time is half the round trip."""
    dev = _device(backend, rank)
    out = []
    for n in sizes:
        x = torch.zeros(max(n // 4, 1), dtype=torch.float32, device=dev)
        peer = 1 - rank

        def once():
            if rank == 0:
                dist.send(x, peer)
                dist.recv(x, peer)
            else:
                dist.recv(x, peer)
                dist.send(x, peer)

        reps = iters if n < 1 << 22 else max(3, iters // 4)
        inner = 10 if n < 1 << 20 else 1  # several round trips per sample, so launch overhead is amortised
        for _ in range(3):
            once()
        sync(x)
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            for _ in range(inner):
                once()
            sync(x)
            ts.append((time.perf_counter() - t0) / inner)
        ts.sort()
        out.append({"bytes": x.numel() * 4, "one_way_s": ts[len(ts) // 2] / 2})
    return out


def allreduce_bench(rank, world, sizes, algos, iters, backend):
    """Time each schedule (and the library's own all_reduce) at each size. Also check the answer once per size."""
    dev = _device(backend, rank)
    use_batch = backend == "nccl"
    rows = []
    for n_bytes in sizes:
        n = max(n_bytes // 4, world)
        base = torch.arange(n, dtype=torch.float32, device=dev) % 7 + rank
        want = (torch.arange(n, dtype=torch.float32, device=dev) % 7) * world + world * (world - 1) / 2
        inner = 10 if n_bytes < 1 << 20 else 1
        for name in algos:
            if name == "library":
                x = base.clone()
                dist.all_reduce(x)
                ok = bool(torch.equal(x, want))
                t = time_op(lambda: dist.all_reduce(x), x, iters, inner=inner)
            else:
                plan = compile_for_rank(schedules.ALLREDUCE[name](world, n), rank)
                x = base.clone()
                execute(plan, x, use_batch=use_batch)
                ok = bool(torch.equal(x, want))
                t = time_op(lambda: execute(plan, x, use_batch=use_batch), x, iters, inner=inner)
            flag = torch.tensor([0 if ok else 1], device=dev)
            dist.all_reduce(flag)
            rows.append({"bytes": n * 4, "algo": name, "seconds": t, "correct": int(flag.item()) == 0})
    return rows


def library_allreduce_sweep(rank, world, sizes, iters):
    """torch.distributed.all_reduce over NCCL at each size; NCCL_PROTO etc. are set by the launcher's env."""
    dev = torch.device("cuda", rank)
    rows = []
    for n_bytes in sizes:
        x = torch.ones(n_bytes // 4, dtype=torch.float32, device=dev)
        inner = 20 if n_bytes < 1 << 24 else 3
        t = time_op(lambda: dist.all_reduce(x), x, iters if n_bytes < 1 << 28 else max(5, iters // 4), inner=inner)
        rows.append({"bytes": n_bytes, "seconds": t})
    return rows


def overlap_probe(rank, world, mm_size, comm_bytes, reps):
    """Time a matmul loop alone, an AllReduce alone, both one after the other, and both at once on two streams."""
    dev = torch.device("cuda", rank)
    a = torch.randn(mm_size, mm_size, device=dev, dtype=torch.bfloat16)
    b = torch.randn(mm_size, mm_size, device=dev, dtype=torch.bfloat16)
    g = torch.ones(comm_bytes // 2, device=dev, dtype=torch.bfloat16)
    comm_stream = torch.cuda.Stream(device=dev)

    def compute():
        for _ in range(reps):
            a @ b

    def comm():
        dist.all_reduce(g)

    def overlapped():
        with torch.cuda.stream(comm_stream):
            dist.all_reduce(g)
        compute()
        torch.cuda.current_stream(dev).wait_stream(comm_stream)

    def compute_timed_under_comm(k):
        """Compute stream time measured with events while k AllReduces run on the other stream."""
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(comm_stream):
            for _ in range(k):  # enough back-to-back AllReduces that the compute loop sits inside communication
                dist.all_reduce(g)
        e0.record()
        compute()
        e1.record()
        torch.cuda.synchronize(dev)
        return e0.elapsed_time(e1) / 1e3

    out = {
        "compute_s": time_op(compute, a, 10),
        "comm_s": time_op(comm, g, 10),
        "sequential_s": time_op(lambda: (comm(), compute()), a, 10),
        "overlapped_s": time_op(overlapped, a, 10),
    }
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    alone = []
    for _ in range(10):
        dist.barrier()
        e0.record()
        compute()
        e1.record()
        torch.cuda.synchronize(dev)
        alone.append(e0.elapsed_time(e1) / 1e3)
    k = int(2 * out["compute_s"] / out["comm_s"]) + 2
    out["allreduces_beside_compute"] = k
    under = []
    for _ in range(10):
        dist.barrier()
        under.append(compute_timed_under_comm(k))
    alone.sort()
    under.sort()
    out["compute_alone_event_s"] = alone[5]
    out["compute_under_comm_event_s"] = under[5]
    out["matmul_tflops"] = 2 * mm_size**3 * reps / alone[5] / 1e12
    return out
