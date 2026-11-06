"""Section 10: coalescing. The sector model (Figure 13) next to measured bandwidth.

GPU: thread i reads element i * stride. Same instruction, same useful bytes; only the stride changes.
CPU: the same strided read with numpy on one thread. The CPU moves 64-byte cache lines, so its cliff is at a
stride of 16 floats instead of 8.

    python labs/why-gpus-are-built-for-deep-learning/scripts/05_coalescing.py --device cuda
    python labs/why-gpus-are-built-for-deep-learning/scripts/05_coalescing.py --device cpu
"""

import time

import _path  # noqa: F401
import numpy as np
import torch

from whygpu import kernels as K
from whygpu.hw import H100_SPEC, Report, bench, device_from_argv
from whygpu.models import warp_sectors

STRIDES = (1, 2, 4, 8, 16, 32)


def gpu(rep: Report) -> None:
    n = 64 << 20  # useful fp32 elements, 256 MiB (long enough that launch and ramp-up do not dominate)
    sms = torch.cuda.get_device_properties().multi_processor_count
    grid, tpb = sms * 8, 256
    out = torch.empty(grid * tpb, device="cuda")
    big = torch.rand(n * max(STRIDES) + 64, device="cuda")
    k = K.get("strided")
    rows = []

    def measure(label, view, stride, model):
        t = bench(lambda: k(grid=(grid, 1, 1), block=(tpb, 1, 1), args=[view, out, stride, n]),
                  "cuda", reps=5, trials=5)
        useful = n * 4 / t / 1e9
        rows.append([label, model["sectors"], f"{100 * model['efficiency']:.1f}%", useful,
                     useful / model["efficiency"], f"{useful / H100_SPEC['hbm_tbps'] / 10:.0f}%"])

    for s in STRIDES:
        measure(f"stride {s}", big, s, warp_sectors(s))
    measure("stride 1, base + 4 bytes", big[1:], 1, warp_sectors(1, misalign_bytes=4))

    g = K.get("gather")
    idx = torch.randint(0, big.numel() - 64, (n,), device="cuda", dtype=torch.int32)
    t = bench(lambda: g(grid=(grid, 1, 1), block=(tpb, 1, 1), args=[big, idx, out, n]), "cuda", reps=5, trials=5)
    sample = warp_sectors(0, indices=idx[:32].tolist())
    useful = n * 4 / t / 1e9
    rows.append(["random gather", sample["sectors"], f"{100 * sample['efficiency']:.1f}%", useful,
                 useful / sample["efficiency"], f"{useful / H100_SPEC['hbm_tbps'] / 10:.0f}%"])
    rep(f"{n:,} useful fp32 loads per run. 'implied GB/s' = useful GB/s / model efficiency.")
    rep("The gather also reads its int32 index array (another 4 bytes per load), not counted as useful.\n")
    rep.table(["pattern", "sectors/warp", "model efficiency", "useful GB/s", "implied GB/s", "useful of spec"], rows)


def cpu(rep: Report) -> None:
    n = 16 << 20  # 64 MiB at stride 1: larger than one CCD's 32 MB L3
    big = np.random.rand(n * max(STRIDES)).astype(np.float32)
    rows = []
    base = None
    for s in STRIDES:
        v = big[: n * s : s]
        v.sum()
        ts = []
        for _ in range(7):
            t0 = time.perf_counter()
            v.sum()
            ts.append(time.perf_counter() - t0)
        t = min(ts)  # the machine is shared; the fastest run is the least disturbed
        gbs = n * 4 / t / 1e9
        base = base or gbs
        line_eff = 1 / min(s, 16)  # a 64-byte line holds 16 floats; a stride of s uses one of every s
        rows.append([f"stride {s}", f"{100 * line_eff:.1f}%", gbs, f"{gbs / base:.2f}"])
    rep(f"numpy, one thread, {n:,} useful fp32 elements, strided view .sum(), fastest of 7.\n")
    rep.table(["pattern", "64B-line efficiency", "useful GB/s", "vs stride 1"], rows)


def main() -> None:
    device = device_from_argv()
    rep = Report(f"05_coalescing_{'h100' if device == 'cuda' else 'cpu'}.txt", device)
    gpu(rep) if device == "cuda" else cpu(rep)
    rep.save()


if __name__ == "__main__":
    main()
