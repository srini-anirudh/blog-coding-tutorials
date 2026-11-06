"""Measure beta: streaming copy bandwidth by buffer size, bytes in flight (Little's law), and launch cost. GPU.

Copy moves 2 bytes across HBM per byte copied (one read, one write); bandwidth below counts both.
"""

import _setup  # noqa: F401
import torch

from rooflab import kernels
from rooflab.hardware import H100, TB
from rooflab.report import save_json, start
from rooflab.timing import auto_time

start("03_bandwidth")
dev = "cuda"
sms = torch.cuda.get_device_properties(0).multi_processor_count
print(f"SMs: {sms}, L2: {torch.cuda.get_device_properties(0).L2_cache_size / 2**20:.0f} MiB, "
      f"spec HBM: {H100.hbm / TB:.2f} TB/s")

print("\n## Copy bandwidth by buffer size (fp32). Small buffers stay in L2 between back-to-back calls.")
print(f"{'buffer':>10} {'torch copy_ TB/s':>17} {'Triton copy TB/s':>17} {'% of spec':>10}")
sizes = [2**k for k in range(20, 33, 1)]  # 1 MiB .. 4 GiB per buffer
sweep = []
for nbytes in sizes:
    n = nbytes // 4
    src = torch.randn(n, device=dev)
    dst = torch.empty_like(src)
    t_torch = auto_time(lambda: dst.copy_(src))
    t_tri = auto_time(lambda: kernels.copy(src, dst, programs=sms * 32))
    bw_t, bw_k = 2 * nbytes / t_torch, 2 * nbytes / t_tri
    sweep.append({"bytes": nbytes, "torch": bw_t, "triton": bw_k})
    print(f"{nbytes / 2**20:8.0f} MiB {bw_t / TB:17.2f} {bw_k / TB:17.2f} {max(bw_t, bw_k) / H100.hbm:10.0%}")
    del src, dst
hbm = max(max(r["torch"], r["triton"]) for r in sweep if r["bytes"] >= 2**30)
print(f"\nHBM bandwidth used as beta from here on: {hbm / TB:.2f} TB/s ({hbm / H100.hbm:.0%} of the 3.35 TB/s spec)")

print("\n## Little's law: fixed 1 GiB copy, grid size limits the bytes in flight")
print("Each program keeps about one BLOCK load outstanding, so bytes in flight ~ programs x BLOCK x 4.")
n = 2**28
src = torch.randn(n, device=dev)
dst = torch.empty_like(src)
block = 4096
little = []
print(f"{'programs':>9} {'in flight KiB':>14} {'TB/s':>7} {'% of beta':>10} {'implied latency ns':>19}")
for progs in (1, 4, 16, 33, 66, 132, 264, 528, 1056, 2112, 4224, 8448):
    t = auto_time(lambda: kernels.copy(src, dst, programs=progs, block=block), budget=0.1)
    bw = 2 * n * 4 / t
    inflight = progs * block * 4
    lat = inflight / bw * 1e9
    little.append({"programs": progs, "inflight": inflight, "bw": bw})
    print(f"{progs:9d} {inflight / 1024:14.0f} {bw / TB:7.3f} {bw / hbm:10.1%} {lat:19.0f}")
print("Implied latency = bytes in flight / bandwidth. It is only meaningful on the rising part, before saturation.")

print("\n## Launch cost: a one-element copy")
a, b = torch.zeros(1, device=dev), torch.zeros(1, device=dev)
t_launch_torch = auto_time(lambda: b.copy_(a))
t_launch_tri = auto_time(lambda: kernels.copy(a, b, programs=1, block=128))
print(f"torch copy_: {t_launch_torch * 1e6:.2f} us per call; Triton copy: {t_launch_tri * 1e6:.2f} us per call "
      f"(back-to-back launches from one Python thread)")

save_json("03_bandwidth", {"hbm_bytes_per_s": hbm, "size_sweep": sweep, "little": little,
                           "launch_s": {"torch": t_launch_torch, "triton": t_launch_tri}})
