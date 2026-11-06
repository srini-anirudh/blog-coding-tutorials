"""Optimizer steps are memory workloads: time follows bytes of state, not FLOPs. GPU, fp32 parameters.

torch's fused SGD, SGD with momentum and AdamW update fp32 tensors only, so the byte counts here leave out the
bf16 weight copy the post includes (12, 20 and 28 bytes per parameter instead of 14, 22 and 30).
Muon's Newton-Schulz step is timed on one square bf16 matrix to show the one optimizer that is compute-bound.
"""

import _setup  # noqa: F401
import torch

from rooflab import model as M
from rooflab.report import measured_ceilings, save_json, start
from rooflab.timing import auto_time

start("09_optimizers")
c = measured_ceilings()
beta, pi = c["beta"], c["bf16"]
print(f"beta = {beta / 1e12:.2f} TB/s ({c['source']})")
dev = "cuda"
numel = 2**28  # 268 M parameters, one fp32 tensor
p = torch.nn.Parameter(torch.randn(numel, device=dev))
p.grad = torch.randn_like(p)

print(f"\n## One optimizer step on {numel / 1e6:.0f} M fp32 parameters")
print(f"{'optimizer':>14} {'B/param':>8} {'model ms':>9} {'measured ms':>12} {'TB/s':>6} {'7B step ms':>11}")
rows = []
for name, make in [
    ("sgd", lambda: torch.optim.SGD([p], lr=1e-3, fused=True)),
    ("sgd_momentum", lambda: torch.optim.SGD([p], lr=1e-3, momentum=0.9, fused=True)),
    ("adam", lambda: torch.optim.AdamW([p], lr=1e-3, fused=True)),
]:
    opt = make()
    opt.step()  # create state
    t = auto_time(opt.step, budget=0.2)
    b = M.optimizer_bytes(name).total
    rows.append({"opt": name, "bytes": b, "t": t, "bw": b * numel / t})
    print(f"{name:>14} {b:8d} {b * numel / beta * 1e3:9.2f} {t * 1e3:12.2f} {b * numel / t / 1e12:6.2f} "
          f"{t / numel * 7e9 * 1e3:11.0f}")
    del opt
    torch.cuda.empty_cache()
print(f"Adam / SGD: bytes {rows[2]['bytes'] / rows[0]['bytes']:.2f}x, time {rows[2]['t'] / rows[0]['t']:.2f}x. "
      "The 7B column scales the measured time linearly; it assumes one GPU holds all the state.")

n = 4096
print(f"\n## Muon's Newton-Schulz on one {n} x {n} bf16 matrix (5 iterations, 3 matmuls each)")
x = torch.randn(n, n, device=dev, dtype=torch.bfloat16)
a_, b_, c_ = 3.4445, -4.7750, 2.0315  # the quintic coefficients from the Muon write-up


def newton_schulz(g):
    y = g / (g.norm() + 1e-7)
    for _ in range(5):
        a = y @ y.T
        y = a_ * y + (b_ * a + c_ * a @ a) @ y
    return y


t_ns = auto_time(lambda: newton_schulz(x), budget=0.2)
w_ns = 5 * 3 * 2 * n**3
print(f"time {t_ns * 1e3:.2f} ms, {w_ns / t_ns / 1e12:.0f} TFLOP/s ({w_ns / t_ns / pi:.0%} of measured bf16 pi). "
      f"Fused AdamW on the same {n * n / 1e6:.1f} M parameters at the measured rate: "
      f"{rows[2]['t'] / numel * n * n * 1e3:.3f} ms.")
save_json("09_optimizers", {"beta": beta, "numel": numel, "steps": rows, "muon_ns_s": t_ns, "muon_n": n})
