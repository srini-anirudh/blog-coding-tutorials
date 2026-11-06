"""Draw the fp32 roofline with one kernel. GPU.

The Triton FMA kernel reads 4 bytes and writes 4 bytes per element and does `iters` FMAs (2 FLOPs each) on it.
So I = 2 iters / 8 = iters / 4, and nothing else changes. The measured points are compared with the overlapped
model min(pi, beta I) and the no-overlap model 1 / (1/pi + 1/(beta I)), using this GPU's own measured beta
(script 03) and fp32 pi (script 04).
"""

import _setup  # noqa: F401
import torch

from rooflab import kernels
from rooflab import model as M
from rooflab.report import measured_ceilings, save_json, start
from rooflab.timing import auto_time

start("05_roofline_sweep")
c = measured_ceilings()
beta, pi = c["beta"], c["fp32"]
print(f"ceilings: beta = {beta / 1e12:.2f} TB/s, fp32 pi = {pi / 1e12:.1f} TFLOP/s ({c['source']}); "
      f"ridge I* = {pi / beta:.1f} FLOP/B")

n = 2**27  # 512 MiB in, 512 MiB out
x = torch.randn(n, device="cuda")
y = torch.empty_like(x)
print(f"\n{'iters':>6} {'I FLOP/B':>9} {'time ms':>8} {'TFLOP/s':>8} {'TB/s':>6} {'roof':>6} {'serial':>7} "
      f"{'% of roof':>10}")
points = []
for iters in (0, 8, 16, 24, 32, 48, 64, 80, 96, 128, 192, 256, 384, 512, 1024, 2048):
    t = auto_time(lambda: kernels.fma_sweep(x, y, iters), budget=0.1)
    i = iters / 4
    w, q = 2 * iters * n, 8 * n
    p = w / t
    roof = M.attainable(i, pi, beta) if i else 0.0
    serial = M.attainable_serial(i, pi, beta) if i else 0.0
    frac = f"{p / roof:10.0%}" if roof else f"{'-':>10}"
    points.append({"iters": iters, "I": i, "t": t, "P": p, "bw": q / t, "roof": roof, "serial": serial})
    print(f"{iters:6d} {i:9.0f} {t * 1e3:8.3f} {p / 1e12:8.2f} {q / t / 1e12:6.2f} {roof / 1e12:6.1f} "
          f"{serial / 1e12:7.1f} {frac}")

near = min(points[1:], key=lambda r: abs(r["I"] - pi / beta))
print(f"\nNearest point to the ridge: I = {near['I']:.0f}, measured {near['P'] / near['roof']:.0%} of the "
      f"overlapped roof; the no-overlap model predicts {near['serial'] / near['roof']:.0%}.")
save_json("05_roofline_sweep", {"beta": beta, "pi_fp32": pi, "n": n, "points": points})
