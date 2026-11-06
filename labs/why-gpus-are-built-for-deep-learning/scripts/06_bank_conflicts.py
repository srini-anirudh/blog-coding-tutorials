"""Section 11: shared memory's 32 banks. The conflict model (Figure 14) next to measured cycles.

Each lane of every warp reads one fixed shared-memory word, over and over. The words come from the same address
patterns as the post's lab: a row, a column of a 32 x 32 tile, the same column with one word of padding, strides,
and one address for everyone. One block of 32 warps per SM keeps the banks busy, so the time is set by bank throughput.

    python labs/why-gpus-are-built-for-deep-learning/scripts/06_bank_conflicts.py
"""

import _path  # noqa: F401
import torch

from whygpu import kernels as K
from whygpu.hw import Report
from whygpu.models import bank_conflict_degree, tile_access

REPS, LOADS_PER_REP, TPB = 1000, 8, 1024


def main() -> None:
    rep = Report("06_bank_conflicts_h100.txt", "cuda")
    k = K.get("banks")
    sms = torch.cuda.get_device_properties().multi_processor_count
    grid = sms  # one block of 32 warps per SM, so one block's cycles are the SM's cycles
    warp_accesses_per_sm = (TPB // 32) * REPS * LOADS_PER_REP
    rep(f"{grid} blocks of {TPB} threads, one per SM. Each lane does {REPS * LOADS_PER_REP:,} shared loads.\n")
    cases = [
        ("row", tile_access("row")),
        ("column, no padding", tile_access("column")),
        ("column, 1 word padding", tile_access("column", pad=1)),
        ("stride 2", tile_access("strided", stride=2)),
        ("stride 4", tile_access("strided", stride=4)),
        ("stride 8", tile_access("strided", stride=8)),
        ("stride 16", tile_access("strided", stride=16)),
        ("stride 3", tile_access("strided", stride=3)),
        ("one address (broadcast)", tile_access("broadcast", col=5)),
    ]
    out = torch.empty(grid * TPB, device="cuda")
    cyc = torch.zeros(grid, dtype=torch.int64, device="cuda")
    rows = []
    for label, addr in cases:
        a = torch.tensor(addr, dtype=torch.int32, device="cuda")
        for _ in range(2):  # the first launch warms up; keep the second
            k(grid=(grid, 1, 1), block=(TPB, 1, 1), args=[a, out, cyc, REPS])
        torch.cuda.synchronize()
        cycles = cyc.float().median().item()
        rows.append([label, bank_conflict_degree(addr), cycles / warp_accesses_per_sm])
    base = rows[0][2]
    for r in rows:
        r.append(r[2] / base)
    rep.table(["access", "model conflict degree", "cycles per warp access (per SM)", "vs row"], rows)
    rep("\nCycles come from clock64 around the loop in each block; the median block is reported.")
    rep.save()


if __name__ == "__main__":
    main()
