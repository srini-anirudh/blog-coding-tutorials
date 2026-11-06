"""Section 8: latency hiding and the occupancy trap.

Part A checks the occupancy model (the post's Figure 11 calculator) against the CUDA driver's own calculator,
cuOccupancyMaxActiveBlocksPerMultiprocessor, for a kernel compiled at several register counts.
Part B is Volkov's point: a streaming read reaches high bandwidth at low occupancy if each thread keeps several
independent loads in flight. Shared memory is requested only to cap how many blocks fit on an SM.

    python labs/why-gpus-are-built-for-deep-learning/scripts/04_occupancy.py
"""

import _path  # noqa: F401
import torch

from whygpu import kernels as K
from whygpu.hw import H100_SPEC, Report, bench
from whygpu.models import CC90, occupancy


def part_a(rep: Report) -> None:
    rep("Part A: model vs driver, blocks per SM")
    rows, agree_new, agree_post, total, seen = [], 0, 0, 0, set()
    for cap in (32, 40, 64, 72, 128, 255):
        k = K.get("hungry", (f"--maxrregcount={cap}",))
        regs = K.num_regs(k)
        if regs in seen:
            continue  # the cap did not bind; this register count is already covered
        seen.add(regs)
        for tpb in (64, 128, 256, 512, 1024):
            if regs * tpb > CC90["regs_per_sm"]:
                continue  # cannot launch at all
            for smem in (0, 10_000, 48 * 1024, 57 * 1024, 100 * 1024, 227 * 1024):
                drv = K.driver_blocks_per_sm(k, tpb, smem)
                ours = occupancy(tpb, regs, smem)["blocks"]
                post = occupancy(tpb, regs, smem, post=True)["blocks"]
                total += 1
                agree_new += ours == drv
                agree_post += post == drv
                if ours != drv or post != drv:
                    rows.append([regs, tpb, smem, drv, ours, post])
    rep(f"{total} configurations. The full model (1 KB reserved per block, registers allocated per processing block) "
        f"matches the driver in {agree_new}. The post's calculator matches in {agree_post}.")
    if rows:
        rep("Configurations where either model differs from the driver:")
        rep.table(["regs", "threads", "smem B", "driver", "full model", "post's calculator"], rows)


def part_b(rep: Report) -> None:
    rep("\nPart B: streaming 1 GiB, bandwidth vs occupancy and loads in flight per thread (ILP)")
    sms = torch.cuda.get_device_properties().multi_processor_count
    n4 = (1 << 30) // 16
    x = torch.rand(n4 * 4, device="cuda")
    tpb = 256
    rows = []
    for ilp in (1, 2, 4, 8):
        k = K.get("stream", ILP=ilp)
        regs = K.num_regs(k)
        for target in (1, 2, 4, 8):
            smem = 0 if target == 8 else (CC90["smem_per_sm"] // target - 1024) // 128 * 128
            blocks = K.driver_blocks_per_sm(k, tpb, smem)
            if blocks != target:
                rows.append([ilp, regs, target, blocks, "-", "-", "skipped: driver gives a different count"])
                continue
            grid = blocks * sms
            out = torch.empty(grid * tpb, device="cuda")
            t = bench(lambda: k(grid=(grid, 1, 1), block=(tpb, 1, 1), args=[x, out, n4], shared_mem=smem),
                      "cuda", reps=5, trials=5)
            gbs = n4 * 16 / t / 1e9
            occ = blocks * tpb / 32 / CC90["warps_per_sm"]
            rows.append([ilp, regs, target, blocks, f"{100 * occ:.1f}%", gbs, f"{gbs / H100_SPEC['hbm_tbps'] / 10:.0f}%"])
    rep.table(["ILP", "regs", "target blocks", "blocks/SM", "occupancy", "GB/s", "of 3.35 TB/s"], rows)
    rep("16-byte loads in flight per SM = blocks x 256 threads x ILP.")


def main() -> None:
    rep = Report("04_occupancy_h100.txt", "cuda")
    part_a(rep)
    part_b(rep)
    rep.save()


if __name__ == "__main__":
    main()
