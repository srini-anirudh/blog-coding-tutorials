"""One number, three machines (Figure 02).

Three kernels run back to back for a few seconds each while a thread samples the NVML utilisation counter
(the number nvidia-smi prints). A: a 4096-element add, which fills a handful of the 132 SMs. B: an
elementwise op on 1 GiB, bandwidth bound. C: an 8192 x 8192 bf16 matmul, compute bound. NVML reports the
fraction of sample periods in which any kernel was resident, so it should read near 100% for all three,
while achieved FLOP/s and bytes/s differ by orders of magnitude.

    python scripts/02_utilisation_lies.py
"""

import sys
import threading
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from whyslow.bench import Report, need_cuda, nvml_utilization  # noqa: E402
from whyslow.roofline import H100_SXM  # noqa: E402


def sample_util(read, stop: threading.Event, out: list, period: float = 0.05):
    while not stop.is_set():
        out.append(read())
        time.sleep(period)


def run_for(fn, seconds: float, per_call_flops: float, per_call_bytes: float, read):
    fn()
    torch.cuda.synchronize()
    stop, utils = threading.Event(), []
    th = threading.Thread(target=sample_util, args=(read, stop, utils)) if read else None
    calls, t0 = 0, time.perf_counter()
    if th:
        th.start()
    while time.perf_counter() - t0 < seconds:
        fn()
        calls += 1
        if calls % 16 == 0:
            torch.cuda.synchronize()  # keep the launch queue short so wall time tracks the GPU
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    if th:
        stop.set()
        th.join()
    warm = utils[len(utils) // 5:] if utils else []  # NVML averages over a window; skip the ramp
    return {"util_pct": sum(warm) / len(warm) if warm else None, "calls": calls, "seconds": dt,
            "tflops": per_call_flops * calls / dt / 1e12, "gbps": per_call_bytes * calls / dt / 1e9}


def main():
    r = Report("02_utilisation_lies")
    if not need_cuda(r):
        return r.save()
    read = nvml_utilization()
    if read is None:
        r("NVML not readable; utilisation column left empty.")

    # A: tiny. Capture 200 launches in one CUDA graph so the CPU is not what keeps the GPU idle.
    small = torch.zeros(4096, device="cuda")
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        small.add_(1.0)
        torch.cuda.synchronize()
        with torch.cuda.graph(g):
            for _ in range(200):
                small.add_(1.0)
    kern_a = g.replay
    a_flops, a_bytes = 200 * 4096.0, 200 * 2 * 4096 * 4.0

    n = 1 << 28  # 256 Mi floats = 1 GiB
    big = torch.randn(n, device="cuda")
    outb = torch.empty_like(big)
    kern_b = lambda: torch.mul(big, 1.0001, out=outb)  # noqa: E731
    b_flops, b_bytes = float(n), 2.0 * 4 * n

    m = 8192
    xa = torch.randn(m, m, device="cuda", dtype=torch.bfloat16)
    xb = torch.randn(m, m, device="cuda", dtype=torch.bfloat16)
    outc = torch.empty(m, m, device="cuda", dtype=torch.bfloat16)
    kern_c = lambda: torch.mm(xa, xb, out=outc)  # noqa: E731
    c_flops, c_bytes = 2.0 * m**3, 3.0 * 2 * m * m

    r(f"{'kernel':42s} {'NVML util':>9s} {'TFLOP/s':>9s} {'GB/s':>8s} {'% bf16 peak':>11s} {'% HBM peak':>10s}")
    rows = {}
    for name, fn, f, b in [("A  add_ on 4096 floats (graph of 200)", kern_a, a_flops, a_bytes),
                           ("B  elementwise on 1 GiB fp32", kern_b, b_flops, b_bytes),
                           ("C  8192^3 bf16 matmul", kern_c, c_flops, c_bytes)]:
        res = run_for(fn, 4.0, f, b, read)
        rows[name] = res
        u = f"{res['util_pct']:8.0f}%" if res["util_pct"] is not None else "      n/a"
        r(f"{name:42s} {u} {res['tflops']:9.3f} {res['gbps']:8.0f} "
          f"{100 * res['tflops'] * 1e12 / H100_SXM['bf16_flops']:10.2f}% {100 * res['gbps'] * 1e9 / H100_SXM['hbm_bytes_per_s']:9.1f}%")
    r("\nPeaks are vendor numbers for H100 SXM (989 TFLOP/s dense bf16, 3.35 TB/s HBM3). Bytes are analytic:")
    r("each operand read once and each output written once.")
    r.data["rows"] = rows
    return r.save()


if __name__ == "__main__":
    main()
