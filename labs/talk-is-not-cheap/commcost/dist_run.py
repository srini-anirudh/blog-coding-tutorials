"""Run a schedule for real with torch.distributed point-to-point calls, on gloo (CPU processes) or NCCL (GPUs).

The same `Transfer` lists that `simulate.run` prices are executed here with isend/irecv, one round at a time. Each
rank only looks at transfers that involve it, so there is no barrier between rounds: a rank waits only for its own
messages, as a real collective library does.
"""

from __future__ import annotations

import os
import socket
import time
from typing import Callable

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from .schedules import Schedule


def compile_for_rank(schedule: Schedule, rank: int):
    """Keep only this rank's sends and receives, round by round. Rounds where it is idle are dropped."""
    plan = []
    for k, rnd in enumerate(schedule):
        sends = [t for t in rnd if t.src == rank and t.dst != rank]
        recvs = [t for t in rnd if t.dst == rank and t.src != rank]
        local = [t for t in rnd if t.src == rank and t.dst == rank]
        if sends or recvs or local:
            plan.append((k, sends, recvs, local))  # k, the global round, is the message tag
    return plan


def execute(plan, x: torch.Tensor, y: torch.Tensor | None = None, use_batch: bool = False) -> None:
    """Run one rank's compiled plan in place on x (and y for alltoall)."""
    bufs = {"x": x, "y": y}
    for i, sends, recvs, local in plan:
        tmp = [torch.empty(t.count, dtype=x.dtype, device=x.device) for t in recvs]
        if use_batch:  # NCCL wants grouped point-to-point to avoid deadlock on symmetric exchanges
            ops = [dist.P2POp(dist.isend, x[t.lo : t.hi], t.dst) for t in sends]
            ops += [dist.P2POp(dist.irecv, b, t.src) for t, b in zip(recvs, tmp)]
            reqs = dist.batch_isend_irecv(ops) if ops else []
        else:
            reqs = [dist.isend(x[t.lo : t.hi], t.dst, tag=i) for t in sends]
            reqs += [dist.irecv(b, t.src, tag=i) for t, b in zip(recvs, tmp)]
        for t in local:
            bufs[t.dst_buf][t.at : t.at + t.count].copy_(x[t.lo : t.hi])
        for r in reqs:
            r.wait()
        for t, b in zip(recvs, tmp):
            view = bufs[t.dst_buf][t.at : t.at + t.count]
            if t.op == "add":
                view.add_(b)
            else:
                view.copy_(b)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _entry(rank, world, port, backend, fn, args, queue, env):
    os.environ.update(env)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.set_num_threads(1)
    if backend == "nccl":
        torch.cuda.set_device(rank)
    dist.init_process_group(backend, rank=rank, world_size=world)
    try:
        out = fn(rank, world, *args)
        if rank == 0:
            queue.put(out)
    finally:
        dist.barrier()
        dist.destroy_process_group()


def launch(world: int, fn: Callable, args=(), backend: str = "gloo", env: dict | None = None):
    """Spawn `world` processes, run fn(rank, world, *args) in each, return rank 0's return value."""
    ctx = mp.get_context("spawn")
    queue = ctx.SimpleQueue()
    mp.start_processes(
        _entry,
        args=(world, free_port(), backend, fn, args, queue, env or {}),
        nprocs=world,
        join=True,
        start_method="spawn",
    )
    return queue.get()


def sync(x: torch.Tensor) -> None:
    if x.is_cuda:
        torch.cuda.synchronize(x.device)


def time_op(op: Callable[[], None], x: torch.Tensor, iters: int, warmup: int = 2, inner: int = 1) -> float:
    """Median seconds per call. Each sample lines all ranks up with a barrier, then times `inner` back-to-back calls
    and divides, so launch and barrier overhead do not swamp small messages."""
    for _ in range(warmup):
        op()
    sync(x)
    ts = []
    for _ in range(iters):
        dist.barrier()
        sync(x)
        t0 = time.perf_counter()
        for _ in range(inner):
            op()
        sync(x)
        ts.append((time.perf_counter() - t0) / inner)
    # every rank must agree, so take the slowest rank's time for each iteration
    tt = torch.tensor(ts, dtype=torch.float64, device=x.device)
    dist.all_reduce(tt, op=dist.ReduceOp.MAX)
    return float(tt.median())
