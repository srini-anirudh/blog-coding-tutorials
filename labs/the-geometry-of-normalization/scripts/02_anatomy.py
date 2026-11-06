"""Sections 03 and 05: the two invariances, the three-term gradient, and the two spheres.

Checks, in order:
  1. Norm(a x) = Norm(x) for LayerNorm and RMSNorm. Norm(x + c 1) = Norm(x) for LayerNorm only.
  2. The hand-derived backward passes match autograd.
  3. The gradient is projected away from 1 (LayerNorm only) and away from xhat (both).
  4. LayerNorm sends points onto a sphere inside the plane sum(x) = 0. RMSNorm uses the whole sphere.
  5. Forward + backward time of torch's LayerNorm and RMSNorm kernels on this CPU.
"""

import argparse
import time

import _setup  # noqa: F401
import torch
import torch.nn.functional as F

from normgeo.ops import layernorm_backward, rmsnorm_backward
from normgeo.report import Report

p = argparse.ArgumentParser()
p.add_argument("--reps", type=int, default=30)
args = p.parse_args()

torch.manual_seed(0)
D, EPS = 64, 1e-5
ln = lambda t: F.layer_norm(t, (t.shape[-1],), eps=EPS)  # noqa: E731
rms = lambda t: F.rms_norm(t, (t.shape[-1],), eps=EPS)  # noqa: E731

with Report("02_anatomy") as say:
    x = torch.randn(512, D, dtype=torch.float64) * 2 + 0.5
    say("1. Invariances, max |Norm(T(x)) - Norm(x)| over 512 vectors of dim 64, float64")
    say(f"   {'transform':<22}{'LayerNorm':>12}{'RMSNorm':>12}")
    for name, t in [("x -> 7.3 x", x * 7.3), ("x -> 0.01 x", x * 0.01), ("x -> x + 3 * 1", x + 3.0)]:
        say(f"   {name:<22}{(ln(t) - ln(x)).abs().max().item():>12.1e}{(rms(t) - rms(x)).abs().max().item():>12.1e}")
    say("   (the 0.01 x row is not exactly zero because eps inside the sqrt starts to matter at small scale)")

    say()
    say("2. Hand-derived backward vs autograd, max abs error, float64")
    gamma = torch.randn(D, dtype=torch.float64)
    dy = torch.randn_like(x)
    for name, fn, back in [("LayerNorm", ln, layernorm_backward), ("RMSNorm", rms, rmsnorm_backward)]:
        xr = x.clone().requires_grad_(True)
        (fn(xr) * gamma * dy).sum().backward()
        say(f"   {name:<12}{(back(x, gamma, dy, EPS) - xr.grad).abs().max().item():.1e}")

    say()
    say("3. Where the gradient points: mean |<dx, u>| / (||dx|| ||u||) over 512 vectors")
    xhat_ln = (x - x.mean(-1, keepdim=True)) / x.std(-1, unbiased=False, keepdim=True)
    xhat_rms = x / x.pow(2).mean(-1, keepdim=True).sqrt()
    ones = torch.ones_like(x)
    cos = lambda a, b: (F.cosine_similarity(a, b, dim=-1).abs().mean().item())  # noqa: E731
    g_ln, g_rms = layernorm_backward(x, gamma, dy, EPS), rmsnorm_backward(x, gamma, dy, EPS)
    say(f"   {'':<12}{'u = 1':>10}{'u = xhat':>10}")
    say(f"   {'LayerNorm':<12}{cos(g_ln, ones):>10.1e}{cos(g_ln, xhat_ln):>10.1e}")
    say(f"   {'RMSNorm':<12}{cos(g_rms, ones):>10.1e}{cos(g_rms, xhat_rms):>10.1e}")
    say(f"   {'input g':<12}{cos(gamma * dy, ones):>10.1e}{cos(gamma * dy, xhat_ln):>10.1e}   (before the projection)")

    say()
    say("4. Geometry. 2000 points in 3D drawn with an offset and unequal scales. eps = 1e-12 here, so only geometry is left.")
    ln = lambda t: F.layer_norm(t, (t.shape[-1],), eps=1e-12)  # noqa: E731
    rms = lambda t: F.rms_norm(t, (t.shape[-1],), eps=1e-12)  # noqa: E731
    cloud = torch.randn(2000, 3, dtype=torch.float64) * torch.tensor([3.0, 1.0, 0.3], dtype=torch.float64) + 1.0
    for name, fn in [("LayerNorm", ln), ("RMSNorm", rms)]:
        out = fn(cloud)
        r, s = out.norm(dim=-1), out.sum(-1)
        say(f"   {name:<10} ||y|| in [{r.min():.4f}, {r.max():.4f}] (sqrt 3 = 1.7321)   x+y+z in [{s.min():+.3f}, {s.max():+.3f}]")
    high = torch.randn(4096, 16, dtype=torch.float64) * 2 + 1
    rank = lambda m: int((torch.linalg.svdvals(m) > 1e-8 * torch.linalg.svdvals(m)[0]).sum())  # noqa: E731
    say(f"   In 16 dims, rank of 4096 outputs: LayerNorm {rank(ln(high))}, RMSNorm {rank(rms(high))}, input {rank(high)}")

    say()
    say(f"5. Forward + backward, (2048 tokens x 1024 features), fp32, median of {args.reps} runs on a shared machine")
    xt = torch.randn(2048, 1024, requires_grad=True)
    w = torch.ones(1024, requires_grad=True)
    times = {}
    for name, fn in [("LayerNorm", lambda: F.layer_norm(xt, (1024,), w)), ("RMSNorm", lambda: F.rms_norm(xt, (1024,), w))]:
        ts = []
        for _ in range(args.reps + 3):
            t0 = time.perf_counter()
            fn().sum().backward()
            ts.append(time.perf_counter() - t0)
        times[name] = sorted(ts[3:])[len(ts[3:]) // 2] * 1e3
        say(f"   {name:<10}{times[name]:7.2f} ms")
    say(f"   RMSNorm / LayerNorm time ratio: {times['RMSNorm'] / times['LayerNorm']:.2f}")
    say("   torch's CPU LayerNorm is one native kernel. Its CPU rms_norm is not faster here, so the saved reduction")
    say("   only pays when the kernel is written to use it. Fused GPU kernels were not measured.")
