"""Section 13: tensor cores, narrow inputs with a wide accumulator, and shapes that do not divide.

1. Accumulation (CPU, numpy, exact and repeatable): a length-4,096 dot product of fp16 values, summed in fp16
   versus summed in fp32, against an fp64 reference. On the GPU, an fp16 matmul with K = 4,096 is compared too.
2. GPU: dense matmul throughput for fp32 (no TF32), TF32, bf16 and fp16, then bf16 at sizes that are and are not
   multiples of the tile shape. CPU: fp32 and bf16 matmul as the baseline.

    python labs/why-gpus-are-built-for-deep-learning/scripts/08_tensor_cores.py --device cuda
    python labs/why-gpus-are-built-for-deep-learning/scripts/08_tensor_cores.py --device cpu
"""

import _path  # noqa: F401
import numpy as np
import torch

from whygpu.hw import H100_SPEC, Report, bench, device_from_argv, set_cpu_threads


def accumulate(k: int = 4096, seed: int = 0) -> dict:
    """Relative error of an fp16 dot product summed in fp16 and in fp32."""
    rng = np.random.default_rng(seed)
    a = rng.random(k).astype(np.float16)
    b = rng.random(k).astype(np.float16)
    exact = float(np.dot(a.astype(np.float64), b.astype(np.float64)))
    s16, s32 = np.float16(0), np.float32(0)
    for x, y in zip(a, b):
        p = np.float16(x * y)  # fp16 product, as an fp16 multiply would give
        s16 = np.float16(s16 + p)
        s32 = np.float32(s32 + np.float32(x) * np.float32(y))
    return {"exact": exact, "fp16_acc": float(s16), "fp32_acc": float(s32),
            "rel_err_fp16": abs(float(s16) - exact) / exact, "rel_err_fp32": abs(float(s32) - exact) / exact}


def tflops(n: int, dtype, device: str, tf32: bool = False) -> float:
    torch.backends.cuda.matmul.allow_tf32 = tf32
    a = torch.randn(n, n, device=device).to(dtype)
    b = torch.randn(n, n, device=device).to(dtype)
    reps = 10 if device == "cuda" else 2
    sec = bench(lambda: a @ b, device, reps=reps, trials=5)
    return 2 * n ** 3 / sec / 1e12


def main() -> None:
    device = device_from_argv()
    rep = Report(f"08_tensor_cores_{'h100' if device == 'cuda' else 'cpu'}.txt", device)
    r = accumulate()
    rep("Length-4,096 dot product of fp16 values in [0, 1):")
    rep(f"  fp64 reference {r['exact']:.4f} | summed in fp16 {r['fp16_acc']:.4f} (rel err {r['rel_err_fp16']:.2e}) "
        f"| summed in fp32 {r['fp32_acc']:.4f} (rel err {r['rel_err_fp32']:.2e})")

    if device == "cpu":
        set_cpu_threads()
        rows = [[dt, 1024, tflops(1024, getattr(torch, dt), "cpu") * 1e3] for dt in ("float32", "bfloat16")]
        rep.table(["dtype", "n", "GFLOP/s"], rows)
        rep.save()
        return

    a = torch.rand(256, 4096, device="cuda", dtype=torch.float16)
    b = torch.rand(4096, 256, device="cuda", dtype=torch.float16)
    exact = a.double() @ b.double()
    got = (a @ b).double()
    rep(f"  GPU fp16 matmul, K = 4,096: max rel err vs fp64 {((got - exact).abs() / exact).max().item():.2e} "
        "(near fp16 output rounding, so the sum was not kept in fp16)")

    rep("\nDense matmul, n = 8192:")
    rows = []
    for label, dt, tf32, peak in (("fp32, TF32 off", torch.float32, False, H100_SPEC["fp32_tflops"]),
                                  ("fp32 inputs, TF32 on", torch.float32, True, 494.5),
                                  ("bf16", torch.bfloat16, False, H100_SPEC["bf16_tflops"]),
                                  ("fp16", torch.float16, False, H100_SPEC["bf16_tflops"])):
        t = tflops(8192, dt, "cuda", tf32)
        rows.append([label, t, peak, f"{100 * t / peak:.0f}%"])
    rep.table(["dtype", "TFLOP/s", "spec dense peak", "of peak"], rows)

    rep("\nbf16 matmul at sizes that do and do not divide the tile shape:")
    rows = [[n, tflops(n, torch.bfloat16, "cuda")] for n in (4096, 4090, 4088, 8192, 8190, 8184)]
    rep.table(["n", "TFLOP/s"], rows)
    rep("Spec peaks: H100 SXM5 published dense figures (TF32 494.5 TFLOP/s dense).")
    rep.save()


if __name__ == "__main__":
    main()
