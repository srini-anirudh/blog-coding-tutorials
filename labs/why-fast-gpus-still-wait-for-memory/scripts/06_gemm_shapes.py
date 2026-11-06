"""Shapes matter: the same d x d weight at different token counts, and square GEMMs by size. GPU, bf16.

For each shape: W = 2MNK, compulsory Q = s(MK + KN + MN), and the roofline floor max(W/pi, Q/beta) with this
GPU's measured beta and bf16 pi. "TB/s" is compulsory bytes over time, so it is a lower bound on real traffic.
"""

import _setup  # noqa: F401
import torch

from rooflab import model as M
from rooflab.report import measured_ceilings, save_json, start
from rooflab.timing import auto_time

start("06_gemm_shapes")
c = measured_ceilings()
beta, pi = c["beta"], c["bf16"]
ridge = pi / beta
print(f"ceilings: beta = {beta / 1e12:.2f} TB/s, bf16 pi = {pi / 1e12:.0f} TFLOP/s ({c['source']}); "
      f"measured ridge I* = {ridge:.0f} FLOP/B")
s, dev = 2, "cuda"


def bench(m, n, k):
    a = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
    b = torch.randn(k, n, device=dev, dtype=torch.bfloat16)
    t = auto_time(lambda: a @ b, budget=0.1)
    w, q = M.gemm_flops(m, n, k), M.gemm_bytes(m, n, k, s)
    return {"M": m, "N": n, "K": k, "t": t, "I": w / q, "P": w / t, "bw": q / t,
            "floor": M.roofline_time(w, q, pi, beta)}


def show(r, label):
    print(f"{label:>7} {r['I']:8.1f} {r['t'] * 1e6:10.1f} {r['floor'] * 1e6:10.1f} {r['P'] / 1e12:8.1f} "
          f"{r['P'] / pi:7.1%} {r['bw'] / 1e12:6.2f} {'compute' if r['I'] > ridge else 'memory':>8}")


hdr = f"{'I':>8} {'time us':>10} {'floor us':>10} {'TFLOP/s':>8} {'of pi':>7} {'TB/s':>6} {'side':>8}"
d = 8192
print(f"\n## One weight, d = {d}, x tokens: y = x @ W  (Table 2 and Figure 08)")
print(f"{'tokens':>7} " + hdr)
proj = []
for m in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 32768):
    r = bench(m, d, d)
    proj.append(r)
    show(r, str(m))
t1 = proj[0]["t"]
flat = max(r["M"] for r in proj if r["t"] < 1.5 * t1)
print(f"\nStep time stays within 1.5x of the 1-token time up to {flat} tokens; "
      f"the roofline model puts the ridge at B* = {s * ridge / 2:.0f}.")
print(f"Batch-1 GEMV moves the weight at {proj[0]['bw'] / 1e12:.2f} TB/s ({proj[0]['bw'] / beta:.0%} of measured beta).")

print("\n## Square GEMMs: compulsory I = n/3 grows with n")
print(f"{'n':>7} " + hdr)
square = []
for n in (256, 512, 768, 1024, 1536, 2048, 4096, 8192, 16384):
    r = bench(n, n, n)
    square.append(r)
    show(r, str(n))
print(f"The compulsory model says a square bf16 GEMM crosses the measured ridge at n = {3 * s * ridge / 2:.0f}.")
save_json("06_gemm_shapes", {"beta": beta, "pi_bf16": pi, "projection": proj, "square": square})
