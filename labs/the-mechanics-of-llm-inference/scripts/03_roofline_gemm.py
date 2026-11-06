"""Post section 3, "The shapes are what matter": one weight matrix, two regimes.

The same d x d bf16 linear layer runs with M = 1 activation row (decode) up to M = 8192 rows (prefill). The script
measures achieved FLOP/s and bytes/s at each M and sets them against the post's intensity formula. It also measures
this GPU's real copy bandwidth and matmul peak, so the ridge point is a measured number, not only a spec.

A pool of distinct weight matrices is cycled so that no matrix is still in the 50 MB L2 cache when it is used
again. Without that, the M = 1 case would read from L2 and look faster than HBM allows.

    python labs/the-mechanics-of-llm-inference/scripts/03_roofline_gemm.py --out .../results/roofline_gemm_h100.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inferlab.ledger import ACCELERATORS, gemm_intensity  # noqa: E402
from inferlab.timing import Report, measured_bandwidth, measured_peak_flops, time_ms  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=4096)
    ap.add_argument("--pool", type=int, default=16, help="distinct weight matrices, cycled to defeat L2")
    ap.add_argument("--rows", default="1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        sys.exit("needs a CUDA GPU")
    rep = Report(args.out)
    spec = ACCELERATORS["h100"]
    bw, peak = measured_bandwidth(), measured_peak_flops()
    rep(f"spec:     {spec.peak_flops / 1e12:.0f} TFLOP/s bf16 dense, {spec.bandwidth / 1e12:.2f} TB/s, "
        f"ridge {spec.peak_flops / spec.bandwidth:.0f} FLOP/byte")
    rep(f"measured: {peak / 1e12:.0f} TFLOP/s (8192^3 bf16 matmul), {bw / 1e12:.2f} TB/s (4 GB copy), "
        f"ridge {peak / bw:.0f} FLOP/byte")
    rep("")
    d = args.d
    weights = [torch.randn(d, d, dtype=torch.bfloat16, device="cuda") * 0.02 for _ in range(args.pool)]
    rep(f"one {d} x {d} bf16 linear layer, weight {d * d * 2 / 2**20:.0f} MiB, pool of {args.pool}")
    rep(f"{'M rows':>7} {'time us':>9} {'TFLOP/s':>8} {'% peak':>7} {'GB/s':>7} {'% bw':>6} {'I (post)':>9}")
    for m in [int(r) for r in args.rows.split(",")]:
        x = torch.randn(m, d, dtype=torch.bfloat16, device="cuda")
        ms = time_ms(lambda: [F.linear(x, w) for w in weights], warmup=3, iters=20) / len(weights)
        flops = 2 * m * d * d
        nbytes = 2 * (m * d + d * d + m * d)
        tf, gbs = flops / (ms / 1e3), nbytes / (ms / 1e3)
        rep(f"{m:>7} {ms * 1e3:>9.1f} {tf / 1e12:>8.1f} {100 * tf / spec.peak_flops:>6.1f}% {gbs / 1e9:>7.0f} "
            f"{100 * gbs / spec.bandwidth:>5.0f}% {gemm_intensity(m, d, d):>9.1f}")
    rep("")
    rep("% peak and % bw are against the spec sheet. I is FLOP per byte counting each operand once at HBM.")
    rep.save()


if __name__ == "__main__":
    main()
