"""Three numbers into a verdict (Figure 04).

For each kernel or phase we take analytic FLOPs, analytic DRAM bytes and a measured time, and read off
the regime: on the sloped roof, on the flat roof, or below both. The cases walk the moves of Figure 04:
a tiny launch-latency-bound matmul, an unfused elementwise chain, the same chain fused by torch.compile
(fewer bytes, same FLOPs), a skinny matmul (decode-like) and a large square matmul.

    python scripts/03_roofline.py
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from whyslow.bench import Report, cuda_time, need_cuda  # noqa: E402
from whyslow.roofline import H100_SXM, elementwise_counts, matmul_counts, time_saved, verdict  # noqa: E402

BF16 = 2


def chain(x, a, b):
    """Four separate elementwise kernels: each reads its input from HBM and writes its output back."""
    y = x * a
    y = y + b
    y = F.gelu(y)
    return y * x


def main():
    r = Report("03_roofline")
    if not need_cuda(r):
        return r.save()
    peak, bw = H100_SXM["bf16_flops"], H100_SXM["hbm_bytes_per_s"]
    cases = []

    t = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    cases.append(("64^3 matmul (one tiny kernel)", *matmul_counts(64, 64, 64, BF16), cuda_time(lambda: t @ t, 200)))

    n = 1 << 26  # 64 Mi elements
    x = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    a, b = torch.randn_like(x), torch.randn_like(x)
    # per-element FLOPs: mul 1, add 1, gelu (erf form) counted as 8, mul 1 -> 11. A convention, not a measurement.
    f_chain = 11.0 * n
    bytes_unfused = BF16 * n * (3 + 3 + 2 + 3)  # (x,a->y) (y,b->y) (y->y) (y,x->y)
    cases.append(("elementwise chain, 4 kernels", f_chain, float(bytes_unfused), cuda_time(lambda: chain(x, a, b))))
    fused = torch.compile(chain)
    fused(x, a, b)
    _, bytes_fused = elementwise_counts(n, 3, 1, 11, BF16)
    cases.append(("same chain, torch.compile (1 kernel)", f_chain, bytes_fused, cuda_time(lambda: fused(x, a, b))))

    w = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    for m in (16, 256):
        h = torch.randn(m, 8192, device="cuda", dtype=torch.bfloat16)
        cases.append((f"{m} x 8192 @ 8192 x 8192 (skinny)", *matmul_counts(m, 8192, 8192, BF16),
                      cuda_time(lambda h=h: h @ w)))
    big = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    cases.append(("8192^3 matmul", *matmul_counts(8192, 8192, 8192, BF16), cuda_time(lambda: big @ w)))

    r(f"Roofs: {peak / 1e12:.0f} TFLOP/s bf16 dense, {bw / 1e12:.2f} TB/s HBM (H100 SXM vendor numbers), "
      f"ridge {peak / bw:.0f} FLOP/byte. FLOPs and bytes are analytic.\n")
    r.data["cases"] = {}
    for name, f, by, sec in cases:
        v = verdict(f, by, sec, peak, bw)
        r(f"{name:38s} {sec * 1e6:9.1f} us  {v.row()}")
        r.data["cases"][name] = vars(v)

    # Measured ceilings: what this card reaches on the friendliest workloads.
    big2 = torch.randn(1 << 28, device="cuda", dtype=torch.bfloat16)
    out2 = torch.empty_like(big2)
    copy_s = cuda_time(lambda: out2.copy_(big2))
    best_mm = max(2 * 8192**3 / cuda_time(lambda: big @ w, 10) for _ in range(3))
    r(f"\nmeasured: copy {2 * 2 * (1 << 28) / copy_s / 1e12:.2f} TB/s (read + write), "
      f"8192^3 bf16 matmul {best_mm / 1e12:.0f} TFLOP/s")
    r.data["measured_copy_tbps"] = 2 * 2 * (1 << 28) / copy_s / 1e12
    r.data["measured_mm_tflops"] = best_mm / 1e12

    r("\nA roofline point needs a time budget (Trap). Fraction of the step saved:")
    for share, speed in [(0.04, 3.0), (0.18, 2.0)]:
        r(f"  part = {share:.0%} of the step, made {speed:g}x faster -> saves {time_saved(share, speed):.1%}")
    return r.save()


if __name__ == "__main__":
    main()
