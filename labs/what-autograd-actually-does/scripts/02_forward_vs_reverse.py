"""Forward mode against reverse mode: counted on paper, then timed on a real network.

1. Multiply-add counts for the two parenthesisations of a Jacobian chain, as the output size grows.
2. The same chain multiplied for real, both orders, to show the value is the same and the time is not.
3. The input gradient of a tanh MLP by reverse mode, forward mode and finite differences, as inputs grow.
4. Finite differences in low precision, and the post's finite-difference cost estimate redone.
"""

import copy
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from autogradlab.chain import chain_cost, multiply_chain, random_jacobians, gradient_three_ways, jvp_vjp_dot_test, mlp  # noqa: E402
from autogradlab.report import Report  # noqa: E402

rep = Report("02_forward_vs_reverse")

rep("1. Jacobian chain n0 = n1 = n2 = 4096 -> n3 outputs. Multiply-adds for each order (exact counts):")
rows = []
for m in [1, 16, 256, 4096, 16384]:
    c = chain_cost([4096, 4096, 4096, m])
    rows.append([m, f"{c['reverse']:.3g}", f"{c['forward']:.3g}", f"{c['forward'] / c['reverse']:.3g}",
                 f"{c['peak_reverse']:.3g}", f"{c['peak_forward']:.3g}"])
rep.table(["outputs", "reverse", "forward", "fwd/rev", "peak rev", "peak fwd"], rows)
rep("At one output, 33.6 million against 68.7 billion, as the post says. At 4096 outputs the orders are close.")

rep("\n2. The same chain multiplied for real (float64, widths 2048 -> 2048 -> 2048 -> 1):")
w = [2048, 2048, 2048, 1]
times = {}
J = random_jacobians(w)
for order in ["reverse", "forward"]:
    multiply_chain(J, order)  # warm-up
    t = time.perf_counter()
    out = multiply_chain(J, order)
    times[order] = time.perf_counter() - t
    times[order + "_value"] = out
diff = (times["reverse_value"] - times["forward_value"]).abs().max().item()
scale = times["reverse_value"].abs().max().item()
rep(f"  reverse order: {times['reverse'] * 1e3:.2f} ms   forward order: {times['forward'] * 1e3:.1f} ms")
rep(f"  largest difference between the two products: {diff:.1e} (largest entry {scale:.1e})")

rep("\n3. Gradient of a scalar MLP output with respect to n inputs (width 256, depth 4, float64):")
rows = []
for n in [16, 64, 256, 1024]:
    model = mlp(n, 256, 4).double()
    x = torch.randn(1, n, dtype=torch.float64)
    gradient_three_ways(model, x)  # warm-up
    r = gradient_three_ways(model, x)
    ref = r["reverse"]["grad"]
    rows.append([n] + [f"{r[k]['seconds'] * 1e3:.1f}" for k in ["reverse", "forward", "finite_diff"]]
                + [f"{(r[k]['grad'] - ref).abs().max().item():.1e}" for k in ["forward", "finite_diff"]])
rep.table(["inputs", "reverse ms", "forward ms", "fin.diff ms", "fwd err", "fd err"], rows)
rep("Reverse mode stays flat as inputs grow. Forward mode and finite differences grow with the number of inputs.")
rep("Forward mode here is vmapped (torch.func.jacfwd), so its n sweeps run as one batched pass.")

W = torch.randn(64, 32, dtype=torch.float64)
a, b = jvp_vjp_dot_test(lambda inp: torch.tanh(inp @ W), torch.randn(4, 64, dtype=torch.float64))
rep(f"\nDot-product test on tanh(x W): <u, J v> = {a:.10f}, <J^T u, v> = {b:.10f}")

rep("\n4. Finite differences in low precision. Error of the central difference against autograd, same model:")
model = mlp(32, 64, 3)
x = torch.randn(1, 32)
rows = []
for dtype in [torch.float64, torch.float32, torch.bfloat16]:
    m = copy.deepcopy(model).to(dtype)
    xx = x.detach().clone().to(dtype).requires_grad_(True)
    m(xx).sum().backward()
    ref = xx.grad.double()
    row = [str(dtype).split(".")[1]]
    for eps in [1e-1, 1e-2, 1e-3, 1e-4]:
        with torch.no_grad():
            g = torch.empty(32, dtype=torch.float64)
            for i in range(32):
                e = torch.zeros_like(xx)
                e[0, i] = eps
                g[i] = (m(xx + e).sum().double() - m(xx - e).sum().double()) / (2 * eps)
        row.append(f"{((g - ref[0]).norm() / ref.norm()).item():.1e}")
    rows.append(row)
rep.table(["dtype", "eps=1e-1", "eps=1e-2", "eps=1e-3", "eps=1e-4"], rows)
rep("Relative error of the whole gradient. In bfloat16 no step size gives a usable answer.")

rep("\nThe post's cost estimate for finite differences, recomputed:")
N = 1e11
flops = 2 * N * (2 * N)  # two forward passes per parameter, 2N FLOPs per token each
seconds = flops / 1e15  # 1000 TFLOP/s
rep(f"  2 passes x {N:.0e} parameters x 2N FLOPs = {flops:.1e} FLOPs per token")
rep(f"  at 1000 TFLOP/s: {seconds:.1e} s = {seconds / 3.156e7:.2f} years per token")
rep("  The post says about 1.3 million years. The same inputs give about 1.3 years per token, so the post's")
rep("  figure looks a factor of 10^6 too large. The conclusion (finite differences are hopeless) still holds.")
rep("  This is arithmetic, not a measurement.")
rep.save()
