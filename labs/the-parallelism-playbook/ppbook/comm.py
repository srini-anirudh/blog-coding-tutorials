"""Collectives with a ledger, and the autograd operators that tensor and sequence parallelism are built from.

Every call goes through `_log`, so a script can count how many collectives of each kind a step issued and
how many bytes each one carried. The counts do not depend on hardware; they follow from where the model was cut.

The four boundary operators (Megatron's names):
  f       identity forward,       AllReduce backward      enters a tensor-parallel region
  f_bar   AllReduce forward,      identity backward       leaves it (row-parallel partial sums)
  g       AllGather(seq) forward, ReduceScatter backward  sequence-parallel region -> TP region
  g_bar   ReduceScatter forward,  AllGather(seq) backward TP region -> sequence-parallel region
"""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager

import torch
import torch.distributed as dist

LEDGER: list[tuple[str, str, int]] = []  # (op, phase, bytes of the input buffer)
_PHASE = ["fwd"]
_FAKE = [False]  # replace every collective by a local no-op, to time the step without communication


def _log(op: str, t: torch.Tensor) -> None:
    LEDGER.append((op, _PHASE[0], t.numel() * t.element_size()))


@contextmanager
def phase(name: str):
    old = _PHASE[0]
    _PHASE[0] = name
    try:
        yield
    finally:
        _PHASE[0] = old


@contextmanager
def fake_comm():
    """Skip the wire. Shapes stay right, values do not: use it only for timing."""
    _FAKE[0] = True
    try:
        yield
    finally:
        _FAKE[0] = False


def reset() -> None:
    LEDGER.clear()


def wire_bytes(op: str, buf: int, n: int) -> float:
    """Bytes one rank sends for one call with a ring algorithm, given the input buffer size."""
    return {"all_reduce": 2 * (n - 1) / n * buf, "reduce_scatter": (n - 1) / n * buf,
            "all_gather": (n - 1) * buf, "all_to_all": (n - 1) / n * buf}.get(op, buf)


def summary(n: int | None = None) -> dict[str, dict]:
    """{op/phase: {"count", "bytes" (input buffers), "wire" (sent per rank)}} for everything since reset()."""
    n = n or (dist.get_world_size() if dist.is_initialized() else 1)
    count, size, wire = Counter(), Counter(), Counter()
    for op, ph, b in LEDGER:
        count[f"{op}/{ph}"] += 1
        size[f"{op}/{ph}"] += b
        wire[f"{op}/{ph}"] += wire_bytes(op, b, n)
    return {k: {"count": count[k], "bytes": size[k], "wire": int(wire[k])} for k in sorted(count)}


# ---- plain collectives (the group defaults to the world) -------------------------------------------

def all_reduce(x: torch.Tensor, group=None) -> torch.Tensor:
    _log("all_reduce", x)
    if not _FAKE[0]:
        dist.all_reduce(x, group=group)
    return x


def all_gather(x: torch.Tensor, dim: int, group=None) -> torch.Tensor:
    _log("all_gather", x)
    n = dist.get_world_size(group)
    if _FAKE[0]:
        return torch.cat([x] * n, dim=dim)
    x = x.movedim(dim, 0).contiguous()
    out = torch.empty((n * x.shape[0],) + x.shape[1:], dtype=x.dtype, device=x.device)
    dist.all_gather_into_tensor(out, x, group=group)
    return out.movedim(0, dim)


def reduce_scatter(x: torch.Tensor, dim: int, group=None) -> torch.Tensor:
    _log("reduce_scatter", x)
    n = dist.get_world_size(group)
    if _FAKE[0]:
        return x.chunk(n, dim=dim)[dist.get_rank(group)].contiguous()
    x = x.movedim(dim, 0).contiguous()
    out = torch.empty((x.shape[0] // n,) + x.shape[1:], dtype=x.dtype, device=x.device)
    dist.reduce_scatter_tensor(out, x, group=group)
    return out.movedim(0, dim)


def all_to_all(x: torch.Tensor, scatter_dim: int, gather_dim: int, group=None) -> torch.Tensor:
    """Split x into n chunks along scatter_dim, send chunk j to rank j, concatenate what arrives
    along gather_dim. A transpose of the layout across ranks."""
    _log("all_to_all", x)
    n = dist.get_world_size(group)
    chunks = [c.contiguous() for c in x.chunk(n, dim=scatter_dim)]
    if _FAKE[0]:
        return torch.cat(chunks, dim=gather_dim) if scatter_dim != gather_dim else x
    flat_in = torch.cat([c.reshape(-1) for c in chunks])  # equal chunks, so one flat buffer works
    flat_out = torch.empty_like(flat_in)
    dist.all_to_all_single(flat_out, flat_in, group=group)
    return torch.cat([p.view_as(c) for p, c in zip(flat_out.chunk(n), chunks)], dim=gather_dim)


# ---- autograd operators ----------------------------------------------------------------------------

class _F(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        return x

    @staticmethod
    def backward(ctx, dy):
        with phase("bwd"):
            return all_reduce(dy.contiguous(), ctx.group), None


class _FBar(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, group):
        return all_reduce(x.contiguous(), group)

    @staticmethod
    def backward(ctx, dy):
        return dy, None


class _G(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim, group):
        ctx.dim, ctx.group = dim, group
        return all_gather(x, dim, group)

    @staticmethod
    def backward(ctx, dy):
        with phase("bwd"):
            return reduce_scatter(dy, ctx.dim, ctx.group), None, None


class _GBar(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim, group):
        ctx.dim, ctx.group = dim, group
        return reduce_scatter(x, dim, group)

    @staticmethod
    def backward(ctx, dy):
        with phase("bwd"):
            return all_gather(dy, ctx.dim, ctx.group), None, None


class _A2A(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scatter_dim, gather_dim, group):
        ctx.dims, ctx.group = (scatter_dim, gather_dim), group
        return all_to_all(x, scatter_dim, gather_dim, group)

    @staticmethod
    def backward(ctx, dy):
        s, g = ctx.dims
        with phase("bwd"):  # the inverse transpose
            return all_to_all(dy, g, s, ctx.group), None, None, None


def f(x, group=None):
    return _F.apply(x, group)


def f_bar(x, group=None):
    return _FBar.apply(x, group)


def g(x, dim=0, group=None):
    return _G.apply(x, dim, group)


def g_bar(x, dim=0, group=None):
    return _GBar.apply(x, dim, group)


def a2a(x, scatter_dim, gather_dim, group=None):
    return _A2A.apply(x, scatter_dim, gather_dim, group)
