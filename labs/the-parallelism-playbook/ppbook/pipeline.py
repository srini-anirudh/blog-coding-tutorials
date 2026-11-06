"""Pipeline schedules, a clock simulator for them, and a real executor that runs one on P processes.

An op is (kind, microbatch, chunk). kind is "F" (forward), "B" (backward; input and weight gradient together)
or, for the zero-bubble schedule, "B" (input gradient only) followed later by "W" (weight gradient).
chunk is the virtual stage on this device (always 0 unless interleaved).
"""

from __future__ import annotations

import time

import torch
import torch.distributed as dist
import torch.nn as nn

Op = tuple[str, int, int]


# ---- schedules: the op order on each stage ----------------------------------------------------------

def gpipe(P: int, m: int) -> list[list[Op]]:
    """All m forwards, then all m backwards."""
    return [[("F", i, 0) for i in range(m)] + [("B", i, 0) for i in range(m)] for _ in range(P)]


def one_f_one_b(P: int, m: int) -> list[list[Op]]:
    """Warm up with P - s - 1 forwards, then alternate one forward and one backward, then drain."""
    out = []
    for s in range(P):
        w = min(P - s - 1, m)
        ops = [("F", i, 0) for i in range(w)]
        for i in range(m - w):
            ops += [("F", w + i, 0), ("B", i, 0)]
        ops += [("B", i, 0) for i in range(m - w, m)]
        out.append(ops)
    return out


def interleaved(P: int, m: int, v: int) -> list[list[Op]]:
    """Megatron's interleaved 1F1B: device s holds virtual stages s, s + P, ..., s + (v - 1) P.
    Microbatches move through the chunks in groups of P. Needs m to be a multiple of P."""
    assert m % P == 0, "interleaved schedule needs m divisible by P"
    total = m * v

    def fwd(k):
        return ("F", (k // (P * v)) * P + k % P, (k % (P * v)) // P)

    def bwd(k):
        return ("B", (k // (P * v)) * P + k % P, v - 1 - (k % (P * v)) // P)

    out = []
    for s in range(P):
        w = min((P - s - 1) * 2 + (v - 1) * P, total)
        ops = [fwd(k) for k in range(w)]
        for k in range(total - w):
            ops += [fwd(w + k), bwd(k)]
        ops += [bwd(k) for k in range(total - w, total)]
        out.append(ops)
    return out


# ---- the clock -----------------------------------------------------------------------------------------

def simulate(P: int, m: int, schedule: str = "1f1b", v: int = 1, t_f: int = 1, t_b: int = 2, t_w: int = 0):
    """Run a schedule on an integer clock. Point-to-point transfers are treated as free (they overlap).

    Costs are per full stage; with v chunks each chunk costs 1 / v of them, so pass costs divisible by v.
    schedule "zb" splits each backward into B (input grad, t_b) and W (weight grad, t_w) and picks greedily:
    B if ready, else F if at most P - s forwards are in flight, else a deferred W (ZB-H1-like; same memory cap).
    Returns busy fraction, bubble ratio, peak live activation stashes per stage, and the timeline.
    """
    V = P * v  # virtual stages; virtual stage j lives on device j % P, chunk j // P
    cost = {"F": t_f // v, "B": t_b // v, "W": t_w // v}
    orders = {"gpipe": lambda: gpipe(P, m), "1f1b": lambda: one_f_one_b(P, m),
              "interleaved": lambda: interleaved(P, m, v)}
    fixed = None if schedule == "zb" else orders[schedule]()
    done: dict[tuple, int] = {}  # (kind, mb, virtual stage) -> finish time
    pos, free_at, live, peak = [0] * P, [0] * P, [0] * P, [0] * P
    timeline = [[] for _ in range(P)]
    todo = {s: {("F", i, c) for i in range(m) for c in range(v)} | {("B", i, c) for i in range(m) for c in range(v)}
            | ({("W", i, 0) for i in range(m)} if schedule == "zb" else set()) for s in range(P)}

    def ready(s, op, t):
        kind, mb, c = op
        j = c * P + s
        if kind == "F":
            dep = ("F", mb, j - 1) if j > 0 else None
        elif kind == "B":
            dep = ("B", mb, j + 1) if j < V - 1 else ("F", mb, j)
        else:
            dep = ("B", mb, j)
        return dep is None or (dep in done and done[dep] <= t)

    t = 0
    while any(todo.values()):
        for s in range(P):
            if free_at[s] > t or not todo[s]:
                continue
            if fixed is not None:
                op = fixed[s][pos[s]]
                if not ready(s, op, t):
                    continue
                pos[s] += 1
            else:
                nxt_f = min((o for o in todo[s] if o[0] == "F"), default=None, key=lambda o: o[1])
                cands = sorted((o for o in todo[s] if o[0] == "B" and ready(s, o, t)), key=lambda o: o[1])
                if not cands and nxt_f and live[s] < P - s and ready(s, nxt_f, t):
                    cands = [nxt_f]
                if not cands:
                    cands = sorted((o for o in todo[s] if o[0] == "W" and ready(s, o, t)), key=lambda o: o[1])
                if not cands:
                    continue
                op = cands[0]
            kind, mb, c = op
            todo[s].discard(op)
            live[s] += {"F": 1, "B": -1, "W": 0}[kind]
            peak[s] = max(peak[s], live[s])
            free_at[s] = t + cost[kind]
            done[(kind, mb, c * P + s)] = free_at[s]
            timeline[s].append((t, free_at[s], op))
        t += 1
    makespan = max(free_at)
    busy = m * (t_f + t_b + t_w)
    return {"makespan": makespan, "bubble": 1 - busy / makespan, "peak_stashes": peak, "timeline": timeline}


def bubble_formula(P: int, m: int, v: int = 1) -> float:
    return (P - 1) / (v * m + P - 1)


def ascii_timeline(sim: dict, width_per_unit: int = 1) -> str:
    rows = []
    for s, ops in enumerate(sim["timeline"]):
        line = [" "] * sim["makespan"] * width_per_unit
        for a, b, (kind, mb, c) in ops:
            ch = {"F": "F", "B": "B", "W": "w"}[kind] if c == 0 else {"F": "f", "B": "b"}[kind]
            for x in range(a * width_per_unit, b * width_per_unit):
                line[x] = ch if x > a * width_per_unit else str(mb % 10)
        rows.append(f"stage {s} |" + "".join(line).replace(" ", ".") + "|")
    return "\n".join(rows)


# ---- the executor: one stage per process, activations over send/recv ----------------------------------

def run_stage(stage: nn.Module, ops: list[Op], inputs, targets, act_shape, groups, dtype=torch.float32):
    """Execute this rank's op list. inputs/targets: lists of m microbatches (used on first/last stage).
    groups: (forward_group, backward_group), so the two directions never queue behind each other.
    Returns peak live stashes, per-op busy seconds and the loss on the last stage."""
    rank, P = dist.get_rank(), dist.get_world_size()
    dev = next(stage.parameters()).device
    m = len(inputs)
    stash, pending, peak, busy, loss_total = {}, [], 0, 0.0, 0.0
    fwd_group, bwd_group = groups
    for kind, mb, _ in ops:
        if kind == "F":
            if rank == 0:
                x = inputs[mb].to(dev)
            else:
                x = torch.empty(act_shape, device=dev, dtype=dtype)
                dist.recv(x, rank - 1, group=fwd_group)
                x.requires_grad_(True)
            _sync(dev)
            t0 = time.perf_counter()
            y = stage(x)
            if rank == P - 1:
                y = nn.functional.mse_loss(y, targets[mb].to(dev)) / m
                loss_total += y.item()
            _sync(dev)
            busy += time.perf_counter() - t0
            if rank < P - 1:
                buf = y.detach().contiguous()  # keep the buffer alive until the send completes
                pending.append((dist.isend(buf, rank + 1, group=fwd_group), buf))
            stash[mb] = (x, y)
            peak = max(peak, len(stash))
        else:
            x, y = stash.pop(mb)
            if rank == P - 1:
                grad = None
            else:
                grad = torch.empty(act_shape, device=dev, dtype=dtype)
                dist.recv(grad, rank + 1, group=bwd_group)
            _sync(dev)
            t0 = time.perf_counter()
            torch.autograd.backward(y, grad)
            _sync(dev)
            busy += time.perf_counter() - t0
            if rank > 0:
                buf = x.grad.contiguous()
                pending.append((dist.isend(buf, rank - 1, group=bwd_group), buf))
    for req, _ in pending:
        req.wait()
    return {"peak_stashes": peak, "busy_s": busy, "loss": loss_total}


def _sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def split_stack(model, P: int, rank: int) -> nn.Sequential:
    """Contiguous split of a Stack: first stage gets the input layer, last stage gets the head."""
    layers = [model.inp, *model.blocks, model.out]
    n = len(model.blocks)
    bounds = [round(i * n / P) for i in range(P + 1)]
    lo, hi = bounds[rank] + (0 if rank == 0 else 1), bounds[rank + 1] + (2 if rank == P - 1 else 1)
    return nn.Sequential(*layers[lo:hi])
