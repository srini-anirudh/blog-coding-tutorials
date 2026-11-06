"""Three Triton kernels, each written to isolate one term of the roofline.

- `copy`: a grid-stride copy whose grid size you choose. Few programs means few bytes in flight (Little's law).
- `fma_sweep`: reads x, does `iters` dependent multiply-adds per element in fp32, writes y. Q is fixed at 8 bytes
  per element and W = 2 * iters, so `iters` slides the kernel along the intensity axis.
- `chain`: residual add -> RMSNorm -> scale by w -> SiLU -> multiply by gate, for one row per program. A launch
  covers stages [start, end); the groups you launch decide which intermediates go through HBM.

Triton is optional. Import this module only on a machine with CUDA and Triton.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _copy_kernel(src, dst, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    progs = tl.num_programs(0)
    nblocks = tl.cdiv(n, BLOCK)
    for b in range(pid, nblocks, progs):
        offs = b * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        tl.store(dst + offs, tl.load(src + offs, mask=m), mask=m)


def copy(src: torch.Tensor, dst: torch.Tensor, programs: int, block: int = 4096, warps: int = 8) -> None:
    n = src.numel()
    _copy_kernel[(programs,)](src, dst, n, BLOCK=block, num_warps=warps)


@triton.jit
def _fma_kernel(x, y, n, iters, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    v = tl.load(x + offs, mask=m)
    a = v
    # 8 dependent FMAs per trip; strict IEEE semantics stop the compiler from collapsing the chain.
    for _ in range(iters // 8):
        for _j in tl.static_range(8):
            a = a * 0.9999 + v
    tl.store(y + offs, a, mask=m)


def fma_sweep(x: torch.Tensor, y: torch.Tensor, iters: int, block: int = 1024) -> None:
    assert x.dtype == torch.float32 and iters % 8 == 0
    n = x.numel()
    _fma_kernel[(triton.cdiv(n, block),)](x, y, n, iters, BLOCK=block, num_warps=4)


NSTAGES = 5
STAGE_NAMES = ["residual +", "RMSNorm", "scale", "SiLU", "gate *"]


@triton.jit
def _chain_kernel(x, r, g, w, t_in, h_out, t_out, d, eps,
                  START: tl.constexpr, END: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    m = cols < d
    base = row * d
    if START == 0:
        v = tl.load(x + base + cols, mask=m, other=0.0).to(tl.float32)
        v += tl.load(r + base + cols, mask=m, other=0.0).to(tl.float32)
        tl.store(h_out + base + cols, v.to(h_out.dtype.element_ty), mask=m)  # the residual stream lives on
    else:
        v = tl.load(t_in + base + cols, mask=m, other=0.0).to(tl.float32)
    if START <= 1 and END > 1:
        v = v * tl.rsqrt(tl.sum(v * v, axis=0) / d + eps)
    if START <= 2 and END > 2:
        v = v * tl.load(w + cols, mask=m, other=0.0).to(tl.float32)
    if START <= 3 and END > 3:
        v = v * tl.sigmoid(v)
    if END > 4:
        v = v * tl.load(g + base + cols, mask=m, other=0.0).to(tl.float32)
    if END > 1:
        tl.store(t_out + base + cols, v.to(t_out.dtype.element_ty), mask=m)


def groups_from_breaks(breaks: list[bool]) -> list[tuple[int, int]]:
    """breaks[i] True means a kernel boundary after stage i (len NSTAGES - 1). Returns [start, end) groups."""
    assert len(breaks) == NSTAGES - 1
    groups, start = [], 0
    for i, b in enumerate(breaks):
        if b:
            groups.append((start, i + 1))
            start = i + 1
    groups.append((start, NSTAGES))
    return groups


def chain(x, r, g, w, breaks: list[bool], eps: float = 1e-6):
    """Run the chain with the given boundaries. Returns (h, y). Intermediates are stored in x.dtype."""
    rows, d = x.shape
    block = triton.next_power_of_2(d)
    h = torch.empty_like(x)
    cur = h
    for start, end in groups_from_breaks(breaks):
        out = torch.empty_like(x) if end > 1 else h
        _chain_kernel[(rows,)](x, r, g, w, cur, h, out, d, eps, START=start, END=end, BLOCK=block,
                               num_warps=8 if block >= 4096 else 4)
        cur = out
    return h, cur


def chain_reference(x, r, g, w, eps: float = 1e-6):
    """Plain PyTorch, fp32 math, for checking the kernel."""
    h = x.float() + r.float()
    v = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
    v = torch.nn.functional.silu(v * w.float()) * g.float()
    return h.to(x.dtype), v.to(x.dtype)
