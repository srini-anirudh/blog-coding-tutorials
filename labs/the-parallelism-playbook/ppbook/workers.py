"""Per-rank functions for `dist_utils.spawn`. Each checks one mechanism against the single-device reference
and returns plain numbers. They live in the package so spawned processes can import them."""

from __future__ import annotations

import time
from contextlib import nullcontext

import torch
import torch.distributed as dist

from . import comm, context_parallel as cp, pipeline as pl
from .dist_utils import device, sync
from .model import Block, Stack, attention
from .ring import ring_all_reduce
from .tensor_parallel import REF_NAMES, TPBlock, mlp_row_first


def _err(a, b) -> float:
    return (a.float() - b.float()).abs().max().item()


# ---- collectives ------------------------------------------------------------------------------------

def collectives_check(numel: int = 1024) -> dict:
    rank, n = dist.get_rank(), dist.get_world_size()
    dev = device()
    x = torch.randn(numel, generator=torch.Generator().manual_seed(rank)).to(dev)
    native = x.clone()
    dist.all_reduce(native)
    ring, sent = ring_all_reduce(x)
    rs = torch.empty(numel // n, device=dev)
    dist.reduce_scatter_tensor(rs, x.clone())
    ag = torch.empty(numel, device=dev)
    dist.all_gather_into_tensor(ag, rs)
    return {"ring_vs_native": _err(ring, native), "rs_then_ag_vs_allreduce": _err(ag, native),
            "bytes_sent_per_rank": sent, "buffer_bytes": numel * 4, "ratio": sent / (numel * 4)}


def bench_collectives(sizes_bytes: list[int], iters: int = 20) -> dict:
    """Time per call for each collective and message size, plus point-to-point send/recv. Like nccl-tests,
    it times `iters` calls back to back and divides, so per-call host synchronisation is not counted."""
    rank, n = dist.get_rank(), dist.get_world_size()
    dev = device()
    out = {}
    for nbytes in sizes_bytes:
        numel = max(n, nbytes // 4 // n * n)
        x = torch.randn(numel, device=dev)
        shard = torch.empty(numel // n, device=dev)
        full = torch.empty(numel, device=dev)
        ops = {"all_reduce": lambda: dist.all_reduce(x),
               "reduce_scatter": lambda: dist.reduce_scatter_tensor(shard, x),
               "all_gather": lambda: dist.all_gather_into_tensor(full, shard),
               "send_recv": lambda: (dist.send(x, 1) if rank == 0 else dist.recv(x, 0)) if rank < 2 else None}
        for name, fn in ops.items():
            for _ in range(3):
                fn()
            sync()
            dist.barrier()
            sync()
            t0 = time.perf_counter()
            for _ in range(iters):
                fn()
            sync()
            out.setdefault(name, []).append((numel * 4, (time.perf_counter() - t0) / iters))
    return out


# ---- tensor and sequence parallelism ------------------------------------------------------------------

def _setup_block(hidden, heads, seq, batch, seed, dtype=torch.float32):
    torch.manual_seed(seed)  # same seed on every rank, so every rank builds the same reference
    dev = device()
    ref = Block(hidden, heads).to(dev, dtype)
    x = torch.randn(seq, batch, hidden, device=dev, dtype=dtype)
    dy = torch.randn(seq, batch, hidden, device=dev, dtype=dtype)
    return ref, x, dy


def _saved_bytes(fn):
    """Bytes autograd keeps for the backward pass while fn runs (the activation memory), deduplicated."""
    seen = {}

    def pack(t):
        seen[(t.data_ptr(), t.shape, t.dtype)] = t.numel() * t.element_size()
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        out = fn()
    return out, sum(seen.values())


def tp_check(hidden=64, heads=8, seq=16, batch=2, sp=False, seed=0, regather=False) -> dict:
    rank, n = dist.get_rank(), dist.get_world_size()
    ref, x, dy = _setup_block(hidden, heads, seq, batch, seed)
    x_ref = x.clone().requires_grad_(True)
    y_ref, ref_saved = _saved_bytes(lambda: ref(x_ref))
    y_ref.backward(dy)

    blk = TPBlock(ref, sequence_parallel=sp, regather=regather)
    x_tp = (x.chunk(n, 0)[rank] if sp else x).clone().requires_grad_(True)
    comm.reset()
    y, tp_saved = _saved_bytes(lambda: blk(x_tp))
    with comm.phase("bwd"):
        y.backward(dy.chunk(n, 0)[rank] if sp else dy)
    blk.sync_replicated_grads()
    ledger = comm.summary()
    grads = blk.full_grads()
    y_full = comm.all_gather(y.detach(), 0) if sp else y.detach()
    dx = comm.all_gather(x_tp.grad, 0) if sp else x_tp.grad
    ref_grads = dict(ref.named_parameters())
    return {"out_err": _err(y_full, y_ref), "dx_err": _err(dx, x_ref.grad),
            "grad_err": max(_err(grads[k], ref_grads[v].grad) for k, v in REF_NAMES.items()),
            "ledger": ledger, "activation_bytes": tp_saved, "ref_activation_bytes": ref_saved,
            "act_msg_bytes": seq * batch * hidden * 4,
            "param_bytes": sum(p.numel() * 4 for p in ref.parameters())}


def row_first_check(hidden=64, seq=16, batch=2, seed=0) -> dict:
    ref, x, _ = _setup_block(hidden, 8, seq, batch, seed)
    with torch.no_grad():
        want = ref.mlp(x)
        comm.reset()
        blk = TPBlock(ref)
        z = comm.f_bar(torch.nn.functional.linear(torch.nn.functional.gelu(
            torch.nn.functional.linear(x, blk.fc1_w, blk.fc1_b)), blk.fc2_w)) + blk.fc2_b
        col_then_row = comm.summary()
        comm.reset()
        got = mlp_row_first(x, ref.fc1.weight, ref.fc1.bias, ref.fc2.weight) + ref.fc2.bias
        row_then_col = comm.summary()
        # skipping the AllReduce before GELU: sum of GELUs of partial sums, which is not GELU of the sum
        rank, n = dist.get_rank(), dist.get_world_size()
        part = torch.nn.functional.linear(x.chunk(n, -1)[rank], ref.fc1.weight.chunk(n, 1)[rank])
        wrong = comm.all_reduce(torch.nn.functional.gelu(part + ref.fc1.bias / n))
        right = torch.nn.functional.gelu(ref.fc1(x))
    return {"col_then_row_err": _err(z, want), "row_then_col_err": _err(got, want),
            "col_then_row_fwd": col_then_row, "row_then_col_fwd": row_then_col,
            "gelu_of_partials_err": _err(wrong, right)}


def bench_tp(hidden, heads, seq, batch, sp=False, regather=False, iters=10, dtype=torch.bfloat16) -> dict:
    """Forward + backward time of one block under TP, with real collectives and with the wire skipped.
    The difference is the communication the step could not hide."""
    rank, n = dist.get_rank(), dist.get_world_size()
    ref, x, dy = _setup_block(hidden, heads, seq, batch, 0, dtype)
    blk = TPBlock(ref, sequence_parallel=sp, regather=regather) if n > 1 else ref
    del ref
    if sp and n > 1:
        x, dy = x.chunk(n, 0)[rank].contiguous(), dy.chunk(n, 0)[rank].contiguous()
    x.requires_grad_(True)

    def step():
        blk(x).backward(dy)

    res = {}
    for mode in ("real", "fake"):
        ctx = comm.fake_comm() if mode == "fake" else nullcontext()
        with ctx:
            for _ in range(3):
                step()
            reps = []
            for _ in range(3):  # three timed loops of `iters` back-to-back steps; keep the median loop
                sync()
                dist.barrier()
                sync()
                t0 = time.perf_counter()
                for _ in range(iters):
                    step()
                sync()
                reps.append((time.perf_counter() - t0) / iters)
        res[mode] = sorted(reps)[1]
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        y = blk(x)
        res["peak_fwd_mem_bytes"] = torch.cuda.max_memory_allocated() - base
        res["held_act_bytes"] = torch.cuda.memory_allocated() - base  # what forward keeps alive for backward
        y.backward(dy)
    res["exposed_comm_s"] = res["real"] - res["fake"]
    return res


# ---- pipeline -------------------------------------------------------------------------------------

def pipeline_run(schedule="1f1b", m=8, hidden=32, heads=4, layers=4, seq=8, micro=2, seed=0,
                 iters=1, dtype=torch.float32) -> dict:
    rank, P = dist.get_rank(), dist.get_world_size()
    dev = device()
    torch.manual_seed(seed)
    ref = Stack(hidden, heads, layers).to(dev, dtype)
    inputs = [torch.randn(seq, micro, 16, device=dev, dtype=dtype) for _ in range(m)]
    targets = [torch.randn(seq, micro, 16, device=dev, dtype=dtype) for _ in range(m)]
    stage = pl.split_stack(ref, P, rank)
    ops = (pl.gpipe if schedule == "gpipe" else pl.one_f_one_b)(P, m)[rank]
    groups = (dist.new_group(list(range(P))), dist.new_group(list(range(P))))
    act_shape = (seq, micro, hidden)
    times, out = [], None
    for it in range(iters + (iters > 1)):  # with repeats, the first step is a warm-up and is not timed
        stage.zero_grad(set_to_none=True)
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats(dev)
        base = torch.cuda.memory_allocated(dev) if dev.type == "cuda" else 0
        sync()
        dist.barrier()
        sync()
        t0 = time.perf_counter()
        out = pl.run_stage(stage, ops, inputs, targets, act_shape, groups, dtype)
        sync()
        dist.barrier()
        if it > 0 or iters == 1:
            times.append(time.perf_counter() - t0)
        if dev.type == "cuda":
            out["peak_mem_bytes"] = torch.cuda.max_memory_allocated(dev) - base
    out["step_s"] = sorted(times)[len(times) // 2]
    out["measured_bubble"] = 1 - out["busy_s"] / out["step_s"]
    if iters == 1 and dtype == torch.float32:  # check the gradients against one device on the whole batch
        mine = {id(p): p.grad.clone() for p in stage.parameters()}
        stage.zero_grad(set_to_none=True)
        loss = sum(torch.nn.functional.mse_loss(ref(x), t) for x, t in zip(inputs, targets)) / m
        loss.backward()
        out["grad_err"] = max(_err(mine[id(p)], p.grad) for p in stage.parameters())
        out["ref_loss"] = loss.item()
    return out


# ---- context parallelism ----------------------------------------------------------------------------

def cp_check(seq=32, heads=4, dim=8, batch=1, layout="contiguous", causal=True, seed=0) -> dict:
    rank, C = dist.get_rank(), dist.get_world_size()
    dev = device()
    torch.manual_seed(seed)
    q, k, v = (torch.randn(batch, heads, seq, dim, device=dev) for _ in range(3))
    want = attention(q, k, v, causal)
    pos = cp.positions(seq, C, rank, layout)
    comm.reset()
    got, blocks = cp.ring_attention(*(cp.shard_seq(t, C, rank, layout) for t in (q, k, v)), pos, layout, causal)
    ring_ledger = comm.summary()
    res = {"ring_err": _err(got, cp.shard_seq(want, C, rank, layout)), "blocks": blocks, "ring_ledger": ring_ledger}
    if layout == "contiguous":  # Ulysses, forward and backward
        qs, ks, vs = (cp.shard_seq(t, C, rank, layout).requires_grad_(True) for t in (q, k, v))
        comm.reset()
        o = cp.ulysses_attention(qs, ks, vs, causal)
        do = torch.randn_like(o)
        o.backward(do)
        res["ulysses_ledger"] = comm.summary()
        qf, kf, vf = (t.clone().requires_grad_(True) for t in (q, k, v))
        full_do = comm.all_gather(do, 2)
        attention(qf, kf, vf, causal).backward(full_do)
        res["ulysses_err"] = _err(o, cp.shard_seq(want, C, rank, layout))
        res["ulysses_grad_err"] = max(_err(a.grad, cp.shard_seq(b.grad, C, rank, layout))
                                      for a, b in ((qs, qf), (ks, kf), (vs, vf)))
    return res


def bench_ring(seq, heads, dim, batch=1, overlap=True, layout="zigzag", iters=5, dtype=torch.bfloat16) -> dict:
    rank, C = dist.get_rank(), dist.get_world_size()
    dev = device()
    q, k, v = (torch.randn(batch, heads, seq // C, dim, device=dev, dtype=dtype) for _ in range(3))
    pos = cp.positions(seq, C, rank, layout)
    times = []
    for i in range(iters + 2):
        sync()
        dist.barrier()
        sync()
        t0 = time.perf_counter()
        _, blocks = cp.ring_attention(q, k, v, pos, layout, True, overlap)
        sync()
        if i >= 2:
            times.append(time.perf_counter() - t0)
    return {"time_s": sorted(times)[len(times) // 2], "blocks": blocks}
