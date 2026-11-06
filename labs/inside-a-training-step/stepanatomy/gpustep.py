"""Measurements that need a CUDA GPU: op timings, the allocator's ledger, step time and MFU, H2D copies."""

from __future__ import annotations

import time
from dataclasses import replace

import torch
import torch.nn.functional as F

from . import accounting as A
from . import flops, models, recompute
from .mixed import MasterWeightAdam

GiB = 2**30


def cuda_ms(fn, iters: int = 20, warmup: int = 3) -> float:
    """Median time of fn() in milliseconds, from CUDA events."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        times.append(a.elapsed_time(b))
    return sorted(times)[len(times) // 2]


def _rmsnorm(x, w, eps=1e-5):
    x32 = x.float()
    return w * (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype)


def _silu_mul(g, u):
    return F.silu(g) * u


def op_table(s: A.Shape, compile_elementwise: bool = True) -> list[dict]:
    """Time each op of one block at shape s in BF16, next to the accounting model's FLOPs and bytes."""
    dev, dt = "cuda", torch.bfloat16
    t, h, i = s.tokens, s.hidden, s.intermediate
    x = torch.randn(t, h, device=dev, dtype=dt)
    w_norm = torch.ones(h, device=dev, dtype=dt)
    w_qkv, w_o = torch.randn(h, 3 * h, device=dev, dtype=dt) * 0.02, torch.randn(h, h, device=dev, dtype=dt) * 0.02
    w_g, w_u = torch.randn(h, i, device=dev, dtype=dt) * 0.02, torch.randn(h, i, device=dev, dtype=dt) * 0.02
    w_d = torch.randn(i, h, device=dev, dtype=dt) * 0.02
    q, k, v = (torch.randn(s.batch, s.heads, s.seq, s.head_dim, device=dev, dtype=dt) for _ in range(3))
    g, u, gi = torch.randn(t, i, device=dev, dtype=dt), torch.randn(t, i, device=dev, dtype=dt), torch.randn(t, i, device=dev, dtype=dt)
    fns = {
        "rmsnorm (attn)": lambda: _rmsnorm(x, w_norm),
        "qkv proj": lambda: x @ w_qkv,
        "attention (flash)": lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True),
        "out proj": lambda: x @ w_o,
        "residual add": lambda: x + x,
        "rmsnorm (mlp)": lambda: _rmsnorm(x, w_norm),
        "gate proj": lambda: x @ w_g,
        "up proj": lambda: x @ w_u,
        "silu * mul": lambda: _silu_mul(g, u),
        "down proj": lambda: gi @ w_d,
    }
    rows = []
    for op in A.block_ops(s):
        ms = cuda_ms(fns[op.name])
        rows.append({"op": op.name, "ms": ms, "TFLOP/s": op.flops / ms / 1e9, "GB/s": op.hbm_bytes / ms / 1e6,
                     "model FLOP/byte": op.intensity, "bound (model)": op.bound})
    if compile_elementwise:
        try:
            c_norm, c_silu = torch.compile(_rmsnorm), torch.compile(_silu_mul)
            for name, fn, op in [("rmsnorm, torch.compile", lambda: c_norm(x, w_norm), A.block_ops(s)[0]),
                                 ("silu * mul, torch.compile", lambda: c_silu(g, u), A.block_ops(s)[8])]:
                ms = cuda_ms(fn)
                rows.append({"op": name, "ms": ms, "TFLOP/s": op.flops / ms / 1e9, "GB/s": op.hbm_bytes / ms / 1e6,
                             "model FLOP/byte": op.intensity, "bound (model)": op.bound})
        except Exception as e:  # no Triton, or a compiler failure: report it, keep the eager rows
            rows.append({"op": f"torch.compile failed: {type(e).__name__}", "ms": float("nan")})
    return rows


def step_report(s: A.Shape, mode: str = "none", steps: int = 5, warmup: int = 2) -> dict:
    """One BF16 TinyLLM at shape s with MasterWeightAdam. Ledger from the allocator, timings from CUDA events."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    # Not built under `with torch.device("cuda")`: that left 0.56 MB of allocator memory live that no tensor
    # accounts for (see the README), which would be mistaken for model weight.
    m = models.build(s, dtype=torch.bfloat16, device="cuda")
    recompute.apply(m, mode)
    n = sum(p.numel() for p in m.parameters())
    # RoPE's cos/sin tables are buffers, not parameters. The model cast them to BF16 along with the weights.
    buffer_bytes = sum(b.numel() * b.element_size() for b in m.buffers())
    param_bytes = sum(p.numel() * p.element_size() for p in m.parameters())
    after_model = torch.cuda.memory_allocated()
    opt = MasterWeightAdam(m.parameters(), lr=3e-4)
    after_opt = torch.cuda.memory_allocated()
    x = models.random_batch(s, device="cuda")

    curve: list[tuple[str, float]] = []
    blocks = m.layers

    def fwd_hook(i):
        def hook(mod, args, out):
            curve.append((f"after forward of block {i}", torch.cuda.memory_allocated() / GiB))
            if out[0].requires_grad:
                out[0].register_hook(lambda gr: curve.append((f"before backward of block {i}",
                                                              torch.cuda.memory_allocated() / GiB)))
        return hook

    times = {"forward": [], "backward": [], "optimizer": []}
    for it in range(warmup + steps):
        record = it == warmup + steps - 1
        handles = [b.register_forward_hook(fwd_hook(i)) for i, b in enumerate(blocks)] if record else []
        if record:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            curve.append(("start of step", torch.cuda.memory_allocated() / GiB))
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
        ev[0].record()
        loss = m(x, labels=x).loss
        ev[1].record()
        if record:
            curve.append(("the turn: loss computed", torch.cuda.memory_allocated() / GiB))
        loss.backward()
        ev[2].record()
        if record:
            curve.append(("after backward", torch.cuda.memory_allocated() / GiB))
            grads_bytes = torch.cuda.memory_allocated() - after_opt
        opt.step()
        ev[3].record()
        torch.cuda.synchronize()
        if record:
            peak = torch.cuda.max_memory_allocated()
            reserved = torch.cuda.memory_reserved()
        opt.zero_grad()
        for h in handles:
            h.remove()
        if it >= warmup:
            times["forward"].append(ev[0].elapsed_time(ev[1]))
            times["backward"].append(ev[1].elapsed_time(ev[2]))
            times["optimizer"].append(ev[2].elapsed_time(ev[3]))
    med = {k: sorted(v)[len(v) // 2] for k, v in times.items()}
    step_ms = sum(med.values())
    n_mm = flops.matmul_params(m)
    useful = 6 * n_mm * s.tokens
    attn = 12 * s.layers * s.hidden * s.seq * s.tokens
    out = {
        "mode": mode, "params": n, "matmul params": n_mm,
        "weights delta bytes": after_model - base, "param bytes": param_bytes, "buffer bytes": buffer_bytes,
        "bytes/param weights (bf16)": (after_model - base) / n,
        "bytes/param master+m+v": (after_opt - after_model) / n,
        "optimizer state delta bytes": after_opt - after_model,
        "bytes/param grads": grads_bytes / n,
        "peak GiB": peak / GiB, "reserved GiB": reserved / GiB,
        "activation+workspace peak GiB": (peak - after_opt) / GiB,
        "ms": med, "step ms": step_ms, "backward / forward": med["backward"] / med["forward"],
        "optimizer share": med["optimizer"] / step_ms,
        "MFU (6ND)": useful / (step_ms / 1e3) / A.H100_BF16_FLOPS,
        "MFU (6ND + attention)": (useful + attn) / (step_ms / 1e3) / A.H100_BF16_FLOPS,
        "MFU fwd+bwd only (6ND)": useful / ((med["forward"] + med["backward"]) / 1e3) / A.H100_BF16_FLOPS,
        "curve": curve,
    }
    del m, opt, x, loss
    torch.cuda.empty_cache()
    return out


def memory_wall(s: A.Shape = A.LLAMA2_7B) -> dict:
    """Try to hold the full 6.7B model's states on one GPU, category by category, until it fails."""
    torch.cuda.empty_cache()
    held, reached = [], []
    total = torch.cuda.get_device_properties(0).total_memory
    n = A.param_count(s)
    try:
        for name, nbytes in A.LEDGER_BYTES.items():
            dtype = torch.bfloat16 if nbytes == 2 else torch.float32
            held.append(torch.empty(n, dtype=dtype, device="cuda"))
            reached.append((name, torch.cuda.memory_allocated() / GiB))
        failed = None
    except torch.OutOfMemoryError:
        failed = list(A.LEDGER_BYTES)[len(reached)]
    out = {"params": n, "gpu GiB": total / GiB, "held": reached, "failed at": failed}
    del held
    torch.cuda.empty_cache()
    return out


def h2d(mb: int = 256) -> dict:
    """Host-to-device copy speed, pageable vs pinned, and whether a copy hides behind a GEMM."""
    n = mb * 2**20
    pageable = torch.empty(n, dtype=torch.uint8).fill_(1)
    pinned = torch.empty(n, dtype=torch.uint8, pin_memory=True).fill_(1)
    dst = torch.empty(n, dtype=torch.uint8, device="cuda")
    out = {}
    for name, src in (("pageable", pageable), ("pinned", pinned)):
        ms = cuda_ms(lambda: dst.copy_(src, non_blocking=True), iters=10)
        out[f"{name} GB/s"] = n / ms / 1e6
    a = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    side = torch.cuda.Stream()

    def gemm():
        return a @ a

    def both(src):
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            dst.copy_(src, non_blocking=True)
        y = gemm()
        torch.cuda.current_stream().wait_stream(side)
        return y

    for name, src in (("pageable", pageable), ("pinned", pinned)):
        t0 = time.perf_counter()
        for _ in range(10):
            both(src)
        torch.cuda.synchronize()
        out[f"gemm + {name} copy, ms"] = (time.perf_counter() - t0) * 100
        t0 = time.perf_counter()
        for _ in range(10):
            dst.copy_(src, non_blocking=True)
        torch.cuda.synchronize()
        out[f"{name} copy alone, ms"] = (time.perf_counter() - t0) * 100
    t0 = time.perf_counter()
    for _ in range(10):
        gemm()
    torch.cuda.synchronize()
    out["gemm alone, ms"] = (time.perf_counter() - t0) * 100
    return out


def post_layer_shape(layers: int) -> A.Shape:
    """The post's per-layer shape (H=4096, I=11008, V=32000, S=4096, B=1) with fewer layers."""
    return replace(A.LLAMA2_7B, layers=layers)
