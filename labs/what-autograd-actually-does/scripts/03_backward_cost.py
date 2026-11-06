"""Why the backward pass costs about twice the forward pass.

1. Count the matmuls that reach the dispatcher: a linear layer, the same layer with no input gradient, and
   attention with and without recomputing P.
2. Count FLOPs with torch's FlopCounterMode for a stack of linear layers: forward against backward.
3. Time forward and backward for the same stack.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.flop_counter import FlopCounterMode  # noqa: E402

from autogradlab.report import Report  # noqa: E402
from autogradlab.vjp import AttentionRecompute, MatmulCounter, attention_forward  # noqa: E402

rep = Report("03_backward_cost")


def count(fn, *inputs):
    with MatmulCounter() as mc:
        out = fn(*inputs)
        mc.phase = "backward"
        out.sum().backward()
    return mc.counts.get("forward", 0), mc.counts.get("backward", 0)


torch.manual_seed(0)
W = torch.randn(64, 32, requires_grad=True)
X = torch.randn(16, 64)
Q, K, V = (torch.randn(2, 128, 32, requires_grad=True) for _ in range(3))
rows = [
    ["linear, input needs grad", *count(lambda x: x @ W, X.clone().requires_grad_(True))],
    ["linear, first layer (input needs no grad)", *count(lambda x: x @ W, X)],
    ["attention, P saved (plain autograd)", *count(lambda q, k, v: attention_forward(q, k, v)[0], Q, K, V)],
    ["attention, P recomputed (FlashAttention style)", *count(AttentionRecompute.apply, Q, K, V)],
]
rep("1. Matmuls that reach the dispatcher:")
rep.table(["operation", "forward", "backward"], rows)
rep("The post counts attention as two matmuls forward and five backward. Plain autograd, which keeps P, needs four.")
rep("The fifth is the QK^T that a recomputing kernel such as FlashAttention runs again in the backward pass.")

torch.manual_seed(0)
width, depth, batch = 2048, 8, 512
model = nn.Sequential(*[nn.Linear(width, width, bias=False) for _ in range(depth)])
x = torch.randn(batch, width)
N = sum(p.numel() for p in model.parameters())

rep(f"\n2. FLOPs for {depth} linear layers of {width} x {width} ({N / 1e6:.1f} M parameters), batch {batch}:")
for label, inp in [("input needs no grad", x), ("input needs grad", x.clone().requires_grad_(True))]:
    with FlopCounterMode(display=False) as fc:
        y = model(inp)
    fwd = fc.get_total_flops()
    with FlopCounterMode(display=False) as fc:
        y.pow(2).mean().backward()
    bwd = fc.get_total_flops()
    rep(f"  {label:<20} forward {fwd / 1e9:7.1f} GFLOP  backward {bwd / 1e9:7.1f} GFLOP  ratio {bwd / fwd:.3f}")
    rep(f"  {'':<20} per token per parameter: forward {fwd / batch / N:.2f}, backward {bwd / batch / N:.2f}")
rep("  The first layer skips dX when the input needs no gradient, so the ratio falls just below 2.")

rep("\n3. Wall time, same stack, median of 7 runs after 2 warm-ups:")
fw, bw = [], []
for i in range(9):
    model.zero_grad(set_to_none=True)
    t0 = time.perf_counter()
    y = model(x)
    loss = y.pow(2).mean()
    t1 = time.perf_counter()
    loss.backward()
    t2 = time.perf_counter()
    if i >= 2:
        fw.append(t1 - t0)
        bw.append(t2 - t1)
f_med, b_med = sorted(fw)[3], sorted(bw)[3]
rep(f"  forward {f_med * 1e3:.1f} ms  backward {b_med * 1e3:.1f} ms  ratio {b_med / f_med:.2f}")
rep("  The machine was shared with other jobs. Treat the times as rough; the FLOP counts above are exact.")
rep.save()
