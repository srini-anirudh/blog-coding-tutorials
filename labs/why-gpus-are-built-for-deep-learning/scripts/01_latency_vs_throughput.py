"""Section 1: latency, throughput and Little's Law, measured.

GPU: the latency of one dependent FMA, then FMA throughput as the number of threads grows. Little's Law predicts
the knee: threads needed = FMA latency (cycles) x FMA lanes on the chip.
Both devices: a chain of tiny dependent ops launched one by one (the GPU loses), and one elementwise op as the
array grows (the GPU wins once it has enough work).

    python labs/why-gpus-are-built-for-deep-learning/scripts/01_latency_vs_throughput.py --device cuda
    python labs/why-gpus-are-built-for-deep-learning/scripts/01_latency_vs_throughput.py --device cpu
"""

import _path  # noqa: F401
import torch

from whygpu import kernels as K
from whygpu.hw import H100_SPEC, Report, bench, device_from_argv, set_cpu_threads
from whygpu.models import littles_law_concurrency


def fma_latency_and_throughput(rep: Report) -> None:
    out = torch.zeros(1 << 20, device="cuda")
    cyc = torch.zeros(32, dtype=torch.int64, device="cuda")
    chain = K.get("fma_chain")
    n = 1_000_000
    chain(grid=(1, 1, 1), block=(1, 1, 1), args=[out, cyc, n, 0.999])
    torch.cuda.synchronize()
    lat = cyc[0].item() / n
    rep(f"One thread, {n:,} dependent fp32 FMAs: {lat:.2f} cycles per FMA (clock64)")

    sms = torch.cuda.get_device_properties().multi_processor_count
    lanes = sms * 128  # 4 processing blocks x 32 fp32 lanes per SM
    need = littles_law_concurrency(lat, lanes)
    rep(f"Little's Law: {lat:.2f} cycles x {lanes:,} lanes ({sms} SMs x 128) = {need:,.0f} FMAs in flight")
    rep("Each thread below runs one dependent chain, so threads in flight = FMAs in flight.\n")

    k = K.get("fma_threads")
    iters = 20_000
    counts = [32, 512, 4224, 16896, 33792, 67584, 135168, 270336, 540672]
    rows, peak = [], 0.0
    for total in counts:
        out = torch.zeros(total, device="cuda")
        block = min(128, total)
        grid = (total + block - 1) // block

        def run():
            k(grid=(grid, 1, 1), block=(block, 1, 1), args=[out, iters, total, 0.999])

        t = bench(run, "cuda", reps=3, trials=5)
        fma_s = total * iters / t
        peak = max(peak, fma_s)
        rows.append([total, f"{total / need:.2f}", t * 1e3, fma_s / 1e12, 2 * fma_s / 1e12])
    for r in rows:
        r.append(f"{100 * r[3] * 1e12 / peak:.0f}%")
    rep.table(["threads", "x Little", "ms", "T FMA/s", "TFLOP/s", "of best"], rows)
    rep(f"\nBest {2 * peak / 1e12:.1f} TFLOP/s fp32 vs {H100_SPEC['fp32_tflops']:.0f} spec "
        "(FMA = 2 FLOP; the spec assumes the boost clock).")


def dependent_ops(rep: Report, device: str) -> None:
    x = torch.ones(1, device=device)

    def chain():
        y = x
        for _ in range(1000):
            y = y * 0.999 + 0.5
        return y

    t = bench(chain, device, reps=3, trials=5) / 2000
    rep(f"\nDependent chain of 2,000 one-element ops launched from Python: {t * 1e6:.2f} us per op")


def elementwise_sweep(rep: Report, device: str) -> None:
    rep("\nOne elementwise op, z = x + y (fp32, 12 bytes moved per element), as the array grows:")
    rows = []
    for p in range(8, 29, 2):
        n = 1 << p
        x = torch.rand(n, device=device)
        y, z = torch.rand(n, device=device), torch.empty_like(x)
        reps = 50 if n < (1 << 22) else 5
        t = bench(lambda: torch.add(x, y, out=z), device, reps=reps, trials=5)
        rows.append([f"2^{p}", t * 1e6, n / t / 1e9, 12 * n / t / 1e9])
    rep.table(["elements", "us", "G elem/s", "GB/s moved"], rows)


def main() -> None:
    device = device_from_argv()
    tag = "h100" if device == "cuda" else "cpu"
    if device == "cpu":
        set_cpu_threads()
    rep = Report(f"01_latency_vs_throughput_{tag}.txt", device)
    if device == "cuda":
        fma_latency_and_throughput(rep)
    dependent_ops(rep, device)
    elementwise_sweep(rep, device)
    rep.save()


if __name__ == "__main__":
    main()
