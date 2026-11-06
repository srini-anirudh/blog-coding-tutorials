"""FlashAttention is not a new attention rule: same numbers, far fewer bytes.
Section "FlashAttention is not a new attention rule". About 30 s on CPU.

The counts come from kernels that really compute attention through a counter on every load and store.
They are element counts in a two-level memory model, not timings. GPU timings: 09_flash_gpu.py.
"""

import _setup  # noqa: F401

from attnmoves.common import Report
from attnmoves.io import reassociation_drift, saved_for_backward, traffic_table

rep = Report("08_flash_io")
d = 64
rep(f"Part 1. Slow-memory traffic per head, d = {d}, in elements (float64 data, so the outputs can be compared)")
rows = []
for r in traffic_table([256, 512, 1024, 2048], d, [64, 128]):
    rows.append([r["n"], r["textbook"], r["textbook formula"], r["tiled B=64"], r["tiled B=64 formula"],
                 r["tiled B=128"], f"{r['textbook'] / r['tiled B=128']:.1f}", r["max |diff| B=128"]])
rep.table(["n", "textbook", "4n^2 + 4nd", "tiled B=64", "formula", "tiled B=128", "textbook / B=128", "max |out diff|"], rows)
rep("Counted traffic equals the formula in every row. Doubling B halves the K and V re-reads.")

n = 8192
rep(f"\nPart 2. One (n, n) intermediate at n = {n} in fp16: {n * n * 2 / 2**20:.0f} MiB per head (computed)")
for k, v in saved_for_backward(n, 128).items():
    rep(f"  saved for backward, d = 128, {k}: {v:,} numbers")

rep("\nPart 3. Same function, different summation order (n = 512, d = 64)")
for k, v in reassociation_drift().items():
    rep(f"  max |diff|, {k}: {v:.2e}")
rep("  Tiled and textbook differ at fp32 rounding level, as the textbook kernel differs from fp64.")
rep.save()
