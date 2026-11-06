"""Measure pi for each instruction mix: fp8, bf16 and tf32 tensor cores, fp32 CUDA cores. GPU.

Large square matmuls (W = 2 n^3) stand in for "every issue slot is an MMA". The fp32 CUDA-core line is measured
twice: an fp32 matmul with TF32 disabled, and the Triton FMA kernel at high intensity.
"""

import _setup  # noqa: F401
import torch

from rooflab import kernels
from rooflab.hardware import H100, H100_TF32, TF
from rooflab.report import save_json, start
from rooflab.timing import auto_time

start("04_compute_ceilings")
dev = "cuda"
n = 8192
w = 2 * n**3
spec = {"fp8": H100.fp8, "bf16": H100.bf16, "tf32": H100_TF32, "fp32": H100.fp32}
ceil = {}

print(f"\n## Square matmul, n = {n}")
print(f"{'pipe':>22} {'TFLOP/s':>9} {'spec':>6} {'% of spec':>10}")


def row(label, key, t, flops=w):
    p = flops / t
    ceil[key] = max(ceil.get(key, 0), p)
    print(f"{label:>22} {p / TF:9.1f} {spec[key] / TF:6.0f} {p / spec[key]:10.1%}")


a16 = torch.randn(n, n, device=dev, dtype=torch.bfloat16)
b16 = torch.randn(n, n, device=dev, dtype=torch.bfloat16)
row("bf16 tensor core", "bf16", auto_time(lambda: a16 @ b16, budget=0.5))

try:  # fp8 e4m3 through the scaled-matmul entry point; B must be column-major
    a8 = a16.to(torch.float8_e4m3fn)
    b8 = b16.t().contiguous().to(torch.float8_e4m3fn).t()
    one = torch.ones((), device=dev)
    row("fp8 e4m3 tensor core", "fp8",
        auto_time(lambda: torch._scaled_mm(a8, b8, scale_a=one, scale_b=one, out_dtype=torch.bfloat16), budget=0.5))
except Exception as e:  # noqa: BLE001
    print(f"{'fp8 e4m3 tensor core':>22} not available here: {type(e).__name__}: {e}")

a32, b32 = a16.float(), b16.float()
torch.backends.cuda.matmul.allow_tf32 = True
row("tf32 tensor core", "tf32", auto_time(lambda: a32 @ b32, budget=0.5))
torch.backends.cuda.matmul.allow_tf32 = False
row("fp32 matmul, no TF32", "fp32", auto_time(lambda: a32 @ b32, budget=0.5))

x = torch.randn(2**26, device=dev)
y = torch.empty_like(x)
iters = 4096
row(f"Triton FMA x{iters}", "fp32", auto_time(lambda: kernels.fma_sweep(x, y, iters), budget=0.5),
    flops=2 * iters * x.numel())

print("\n## The cliff between tensor cores and CUDA cores")
print(f"bf16 tensor core / fp32 CUDA core: measured {ceil['bf16'] / ceil['fp32']:.1f}x, "
      f"spec {H100.bf16 / H100.fp32:.1f}x")
save_json("04_compute_ceilings", {"n": n, "ceilings": ceil, "spec": spec})
