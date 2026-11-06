"""Sections 1, 4, 5, 11 and 14 on real GPUs: the ladder's rungs, NCCL protocols, and the cost of overlap.

    CUDA_VISIBLE_DEVICES=0,1 python scripts/09_gpu_links.py

Needs 2 GPUs. Measures, on this machine:
  1. HBM: device-to-device copy bandwidth on one GPU.
  2. PCIe: host-to-device and device-to-host copy bandwidth, pinned and pageable.
  3. NVLink: peer-to-peer copy between the two GPUs (whether P2P is enabled is reported).
  4. NCCL AllReduce bus bandwidth against size, with NCCL_PROTO forced to LL, LL128 and Simple (nuance 08).
  5. Overlap: a bf16 matmul loop alone, an AllReduce alone, the two in sequence, the two on separate streams, and
     how much the matmul slows down while an AllReduce runs beside it (nuance 15).
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from commcost import alphabeta as ab  # noqa: E402
from commcost import dist_run, workers  # noqa: E402
from commcost.report import Report, human_bytes, table  # noqa: E402


def sync_all():
    for i in range(torch.cuda.device_count()):
        torch.cuda.synchronize(i)


def copy_bw(dst, src, reps=10):
    """Copy bandwidth in bytes/s. Synchronises every GPU, since a copy to GPU 1 does not run on GPU 0's stream."""
    gpu = dst if dst.is_cuda else src
    with torch.cuda.device(gpu.device):
        for _ in range(2):
            dst.copy_(src, non_blocking=True)
        sync_all()
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            dst.copy_(src, non_blocking=True)
            sync_all()
            ts.append(time.perf_counter() - t0)
    ts.sort()
    return src.numel() * src.element_size() / ts[len(ts) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--copy-bytes", type=int, default=1 << 30)
    ap.add_argument("--quick", action="store_true", help="smaller sizes, for a smoke test")
    args = ap.parse_args()
    assert torch.cuda.device_count() >= 2, "needs two visible GPUs"
    R = Report("09_gpu_links", gpu=True, note="2 GPUs visible")
    n = args.copy_bytes if not args.quick else 1 << 26
    d0, d1 = torch.device("cuda", 0), torch.device("cuda", 1)
    data = {}

    R("1-3. Copy bandwidth, GB/s, median of 10, one direction")
    src = torch.empty(n, dtype=torch.uint8, device=d0)
    rows = []
    hbm = copy_bw(torch.empty_like(src), src) * 2  # a copy reads n and writes n
    rows.append(["HBM, device to device on GPU 0 (read + write)", f"{hbm / 1e9:,.0f}"])
    for dev in (d0, d1):
        on = src if dev == d0 else torch.empty(n, dtype=torch.uint8, device=d1)
        for pinned in (True, False):
            h = torch.empty(n, dtype=torch.uint8, pin_memory=pinned)
            h.fill_(1)  # touch every page before timing
            kind = "pinned" if pinned else "pageable"
            rows.append([f"PCIe host to GPU {dev.index}, {kind}", f"{copy_bw(on, h) / 1e9:.1f}"])
            rows.append([f"PCIe GPU {dev.index} to host, {kind}", f"{copy_bw(h, on) / 1e9:.1f}"])
    p2p = torch.cuda.can_device_access_peer(0, 1)
    peer = copy_bw(torch.empty(n, dtype=torch.uint8, device=d1), src)
    rows.append([f"GPU 0 to GPU 1 copy (peer access {'on' if p2p else 'off'})", f"{peer / 1e9:.1f}"])
    for line in table(rows, ["path", "GB/s"]):
        R(line)
    data["copies_GBps"] = {r[0]: float(r[1].replace(",", "")) for r in rows}
    R(f"Transfer size {human_bytes(n)}. Spec values from the post: HBM3 3,350, PCIe Gen5 x16 63, NVLink 4 450.")
    del src
    torch.cuda.empty_cache()

    R()
    R("4. NCCL AllReduce over 2 GPUs, bus bandwidth GB/s (nccl-tests convention, 2(p-1)/p = 1 at p = 2)")
    top = 30 if not args.quick else 24
    sizes = [1 << k for k in range(12, top + 1, 2)]
    sweeps = {}
    for proto in ("LL", "LL128", "Simple"):
        rows_p = dist_run.launch(2, workers.library_allreduce_sweep, (sizes, 20), backend="nccl",
                                 env={"NCCL_PROTO": proto})
        sweeps[proto] = rows_p
    out = []
    for i, s in enumerate(sizes):
        out.append([human_bytes(s)] + [
            f"{ab.bus_bandwidth(s, sweeps[pr][i]['seconds'], 2) / 1e9:.1f} ({sweeps[pr][i]['seconds'] * 1e6:.0f} us)"
            for pr in sweeps])
    for line in table(out, ["n", *[f"{pr} GB/s (time)" for pr in sweeps]]):
        R(line)
    data["nccl_proto"] = sweeps

    R()
    R("5. Overlap on 2 GPUs: bf16 8192^3 matmuls x 20 and a 1 GiB AllReduce")
    ov = dist_run.launch(2, workers.overlap_probe, (8192 if not args.quick else 2048, 1 << 30 if not args.quick
                                                     else 1 << 24, 20), backend="nccl")
    R(f"matmul loop alone:       {ov['compute_s'] * 1e3:8.1f} ms  ({ov['matmul_tflops']:.0f} TFLOP/s)")
    R(f"AllReduce alone:         {ov['comm_s'] * 1e3:8.1f} ms")
    R(f"one after the other:     {ov['sequential_s'] * 1e3:8.1f} ms")
    R(f"on separate streams:     {ov['overlapped_s'] * 1e3:8.1f} ms")
    slow = ov["compute_under_comm_event_s"] / ov["compute_alone_event_s"] - 1
    R(f"matmul loop by CUDA events, alone {ov['compute_alone_event_s'] * 1e3:.1f} ms, beside an AllReduce "
      f"{ov['compute_under_comm_event_s'] * 1e3:.1f} ms: {slow:+.1%}")
    data["overlap"] = ov
    R.save(data)


if __name__ == "__main__":
    main()
