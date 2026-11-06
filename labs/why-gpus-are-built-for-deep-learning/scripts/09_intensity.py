"""Section 14: when more compute stops helping. Figure 17, measured.

Each transformer op is timed on its own. Its FLOPs and minimum bytes come from whygpu.models.op_cost (the same
hand counts as the post). The ridge point is peak FLOP/s / bandwidth: 989 / 3.35 = 295 FLOP per byte on H100.
An op below the ridge cannot use the tensor cores' rate however much parallelism it has.

    python labs/why-gpus-are-built-for-deep-learning/scripts/09_intensity.py --device cuda
    python labs/why-gpus-are-built-for-deep-learning/scripts/09_intensity.py --device cpu
"""

import _path  # noqa: F401
import torch
import torch.nn.functional as F

from whygpu.hw import H100_SPEC, Report, bench, device_from_argv, set_cpu_threads
from whygpu.models import op_cost, ridge_point


def main() -> None:
    device = device_from_argv()
    gpu = device == "cuda"
    if not gpu:
        set_cpu_threads()
    rep = Report(f"09_intensity_{'h100' if gpu else 'cpu'}.txt", device)
    dtype = torch.float16 if gpu else torch.float32
    eb = 2 if gpu else 4
    rows_n = 8192 if gpu else 2048
    x = torch.randn(rows_n, 8192, device=device, dtype=dtype)
    y = torch.randn_like(x)
    out = torch.empty_like(x)
    mm_n = 8192 if gpu else 2048
    a = torch.randn(mm_n, mm_n, device=device, dtype=dtype)
    b = torch.randn_like(a)

    ops = [
        ("copy", lambda: out.copy_(x), 0, 2 * x.numel() * eb),
        ("elementwise add", lambda: torch.add(x, y, out=out), *op_cost("add", x.numel(), eb)),
        ("GELU (tanh)", lambda: F.gelu(x, approximate="tanh"), *op_cost("gelu", x.numel(), eb)),
        ("LayerNorm", lambda: F.layer_norm(x, (x.shape[-1],)), *op_cost("layernorm", x.numel(), eb)),
        ("softmax", lambda: torch.softmax(x, -1), *op_cost("softmax", x.numel(), eb)),
        (f"matmul {mm_n}^3", lambda: a @ b, *op_cost("matmul", 0, eb, n=mm_n)),
    ]
    if gpu:
        peak, bw = H100_SPEC["bf16_tflops"] * 1e12, H100_SPEC["hbm_tbps"] * 1e12
        rep(f"Ridge point: {peak / 1e12:.0f} TFLOP/s / {bw / 1e12} TB/s = {ridge_point(peak, bw):.0f} FLOP/byte (spec).")
    rep(f"Tensors {rows_n} x 8192, {dtype}. Bytes are the minimum: read inputs once, write outputs once.\n")
    rows = []
    for name, fn, flops, nbytes in ops:
        sec = bench(fn, device, reps=10 if gpu else 2, trials=5)
        ai = flops / nbytes
        row = [name, f"{ai:.2f}", sec * 1e3, flops / sec / (1e12 if gpu else 1e9), nbytes / sec / 1e9]
        if gpu:
            row += [f"{100 * flops / sec / peak:.1f}%", f"{100 * nbytes / sec / bw:.0f}%"]
        rows.append(row)
    head = ["op", "FLOP/byte", "ms", "TFLOP/s" if gpu else "GFLOP/s", "GB/s"]
    rep.table(head + (["of 989 TFLOP/s", "of 3.35 TB/s"] if gpu else []), rows)
    rep.save()


if __name__ == "__main__":
    main()
