"""Count GEMM traffic under three accountings: compulsory, tiled with no cache, and tiled with an LRU cache.

The cache is a stand-in for L2. The point is the post's Figure 06 and 07: the tile size sets the no-cache
intensity, and a cache absorbs re-reads only if the tile order keeps neighbours together. CPU only.
"""

import argparse

import _setup  # noqa: F401

from rooflab import model as M
from rooflab.report import start
from rooflab.tilesim import simulate

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=4096)
p.add_argument("--wave", type=int, default=132, help="tiles in flight at once (H100 has 132 SMs)")
args = p.parse_args()
start("02_tile_traffic")

n, s = args.n, 2
w = M.gemm_flops(n, n, n)
q_comp = M.gemm_bytes(n, n, n, s)
print(f"\nM = N = K = {n}, bf16. W = {w / 1e9:.1f} GFLOP. Q_compulsory = {q_comp / 1e6:.1f} MB, "
      f"I = {w / q_comp:.0f}")

print("\n## Tile size sets the no-cache intensity (formula vs counted)")
print(f"{'tile':>8} {'Q formula MB':>13} {'Q counted MB':>13} {'I':>6} {'limit T/s':>10}")
for t in (32, 64, 128, 256):
    q_f = M.gemm_bytes_tiled(n, n, n, s, t, t)
    q_c = simulate(n, n, n, s, t, t, 64, cache_bytes=0).hbm_bytes
    print(f"{t:>4}x{t:<3} {q_f / 1e6:13.1f} {q_c / 1e6:13.1f} {w / q_c:6.1f} {t / s:10.0f}")

print(f"\n## A cache absorbs re-reads if the order lets it. Tile 128x128, Tk 64, {args.wave} tiles in flight")
print(f"{'cache':>8} {'order':>8} {'Q MB':>9} {'hit rate':>9} {'I':>7} {'x compulsory':>13}")
for cache_mb in (0, 1, 4, 50):
    for order in ("row", "grouped"):
        r = simulate(n, n, n, s, 128, 128, 64, cache_bytes=cache_mb * 2**20, order=order, wave=args.wave)
        print(f"{cache_mb:>5} MB {order:>8} {r.hbm_bytes / 1e6:9.1f} {r.hit_rate:9.2%} "
              f"{w / r.hbm_bytes:7.0f} {r.hbm_bytes / q_comp:13.2f}")
print("\nThe LRU model is far simpler than real L2 (sectored, partitioned, not LRU). It shows the direction only.")
