"""Two consequences of the += in the adjoint recurrence: rounding depends on order, and small gradients underflow.

1. Sum the same gradient contributions in different orders. float32 addition is not associative.
2. Scale the loss by 2^16, backpropagate, and divide out: the gradient is bitwise the same, because
   differentiation is linear and power-of-two scaling is exact.
3. Gradients of size 1e-8 underflow to zero in float16 and survive with loss scaling. bfloat16 has the exponent
   range of float32 and does not need it.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from autogradlab.report import Report  # noqa: E402

rep = Report("09_precision_and_order")
torch.manual_seed(0)

rep("1. One node receives 4096 gradient contributions. Sum them in 20 random orders:")
contrib = torch.randn(4096, dtype=torch.float64) * torch.logspace(-3, 3, 4096, dtype=torch.float64)
rows = []
for dtype in [torch.float32, torch.float64]:
    c = contrib.to(dtype)
    sums = []
    for i in range(20):
        perm = torch.randperm(4096, generator=torch.Generator().manual_seed(i))
        acc = torch.zeros((), dtype=dtype)
        for v in c[perm]:
            acc += v  # the adjoint +=, one child at a time
        sums.append(acc.item())
    exact = contrib.sum().item()
    spread = max(sums) - min(sums)
    rows.append([str(dtype).split(".")[1], len(set(sums)), f"{spread:.2e}", f"{spread / abs(exact):.1e}"])
rep.table(["dtype", "distinct results", "max - min", "relative spread"], rows)
rep("Same numbers, same machine, different order: different bits. Engines that schedule children on several")
rep("threads inherit this. Deterministic modes fix the order, and pay for it in throughput.")

rep("\n2. Loss scaling is exact for powers of two (float32 MLP, one backward):")
model = torch.nn.Sequential(torch.nn.Linear(32, 64), torch.nn.Tanh(), torch.nn.Linear(64, 1))
data = torch.randn(16, 32)
for scale in [1.0, 2.0**16, 1000.0]:
    model.zero_grad()
    (model(data).pow(2).mean() * scale).backward()
    g = torch.cat([p.grad.reshape(-1) for p in model.parameters()]) / scale
    if scale == 1.0:
        g0 = g.clone()
    rep(f"  scale {scale:>8g}: gradients equal to unscaled bit for bit: {torch.equal(g, g0)}; "
        f"max abs diff {(g - g0).abs().max().item():.1e}")

rep("\n3. Gradients near 1e-8 in low precision (loss = 1e-4 * sum(w * x), x ~ 1e-4, so dL/dw ~ 1e-8):")
x32 = torch.rand(1000) * 2e-4
rows = []
for dtype, scale in [(torch.float16, 1.0), (torch.float16, 2.0**12), (torch.float16, 2.0**16), (torch.bfloat16, 1.0)]:
    w = torch.ones(1000, dtype=dtype, requires_grad=True)
    loss = (w * x32.to(dtype)).sum() * 1e-4
    (loss * scale).backward()
    g = w.grad.float() / scale
    true = x32 * 1e-4
    rows.append([str(dtype).split(".")[1], f"{scale:g}", f"{(g == 0).float().mean().item():.1%}",
                 f"{((g - true).norm() / true.norm()).item():.1e}"])
rep.table(["dtype", "loss scale", "grads that are 0", "relative error"], rows)
rep("float16's smallest subnormal is about 6e-8, so these gradients all become 0. Scaling the loss by 2^12 lifts")
rep("them into range. 2^16 = 65536 is above float16's largest value (65504), so the scaled seed is already inf: this")
rep("is why loss scalers lower the scale when they see an inf. bfloat16 keeps the gradients without any scaling.")
rep.save()
