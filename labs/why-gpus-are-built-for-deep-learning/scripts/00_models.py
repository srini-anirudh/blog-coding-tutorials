"""The post's interactive labs as tables, from whygpu.models. No GPU, no timing: these are counts.

    python labs/why-gpus-are-built-for-deep-learning/scripts/00_models.py
"""

import _path  # noqa: F401

from whygpu.hw import H100_SPEC, Report
from whygpu.models import (assign_paths, bank_conflict_degree, divergence_passes, occupancy, ridge_point,
                           simd_loop, tile_access, warp_sectors)


def main() -> None:
    rep = Report("00_models.txt", "cpu")

    rep("Figure 05, vectorising a loop (n = 37 and n = 1024):")
    rows = []
    for n in (37, 1024):
        for name, w in (("scalar", 1), ("SSE", 4), ("AVX2", 8), ("AVX-512", 16)):
            m = simd_loop(n, w)
            rows.append([n, name, m["instructions"], m["wasted_lanes"], f"{100 * m['utilisation']:.0f}%",
                         m["speedup_vs_scalar"]])
    rep.table(["n", "width", "vector instr", "wasted lanes", "utilisation", "speed-up"], rows)

    rep("\nFigure 09, divergence: 256 threads, slowdown = passes / passes without divergence:")
    rows = []
    for p in (1, 2, 4, 8):
        for mode in ("warp", "tid", "random"):
            rows.append([p, mode, divergence_passes(assign_paths(256, p, mode)) / 8])
    rep.table(["paths", "mapping", "slowdown"], rows)

    rep("\nFigure 11, occupancy (CC 9.0). The full model adds the 1 KB per-block reserve and per-partition registers:")
    rows = []
    for tpb, regs, smem in ((256, 32, 0), (256, 64, 0), (256, 128, 0), (256, 255, 0), (64, 102, 0),
                            (128, 32, 57 * 1024), (256, 32, 100 * 1024)):
        o, p = occupancy(tpb, regs, smem), occupancy(tpb, regs, smem, post=True)
        rows.append([tpb, regs, smem, p["blocks"], o["blocks"], f"{100 * o['occupancy']:.0f}%", ", ".join(o["binding"])])
    rep.table(["threads", "regs", "smem B", "post's blocks", "full model blocks", "occupancy", "binding"], rows)

    rep("\nFigure 13, one warp's fp32 load:")
    rows = []
    for s in (1, 2, 4, 8, 16, 32):
        m = warp_sectors(s)
        rows.append([f"stride {s}", m["sectors"], m["bytes_moved"], f"{100 * m['efficiency']:.1f}%"])
    m = warp_sectors(1, misalign_bytes=4)
    rows.append(["stride 1, +4 B", m["sectors"], m["bytes_moved"], f"{100 * m['efficiency']:.1f}%"])
    rep.table(["pattern", "sectors", "bytes moved", "efficiency"], rows)

    rep("\nFigure 14, 32 x 32 float tile in shared memory:")
    rows = [[lbl, bank_conflict_degree(a)] for lbl, a in (
        ("row", tile_access("row")), ("column", tile_access("column")),
        ("column, pad 1", tile_access("column", pad=1)), ("stride 2", tile_access("strided", stride=2)),
        ("stride 16", tile_access("strided", stride=16)), ("stride 3", tile_access("strided", stride=3)),
        ("broadcast", tile_access("broadcast")))]
    rep.table(["access", "conflict degree"], rows)

    rid = ridge_point(H100_SPEC["bf16_tflops"] * 1e12, H100_SPEC["hbm_tbps"] * 1e12)
    rep(f"\nRidge point, H100 SXM5 spec: {rid:.0f} FLOP per byte")
    rep.save()


if __name__ == "__main__":
    main()
