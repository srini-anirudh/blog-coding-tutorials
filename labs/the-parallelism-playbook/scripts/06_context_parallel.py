"""Context parallelism: ring attention with online softmax is exact; causal masking makes contiguous shards
uneven and the zigzag layout evens them; Ulysses transposes the layout with All-to-All and runs plain attention.
With --bench it times the ring with and without posting the next KV exchange before the current block."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ppbook import workers  # noqa: E402
from ppbook.dist_utils import Report, spawn  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--world", type=int, default=2)
    args = ap.parse_args()
    if args.bench:
        rep = Report(f"06_ring_bench_C{args.world}.txt")
        rep(f"ring attention, causal, zigzag layout, {args.world} ranks, bf16, batch 1, 16 heads, head dim 128;"
            " median of 5. This ring materialises each block's scores in fp32 (no fused kernel).")
        rep(f"{'seq':>7} {'overlap ms':>11} {'no overlap ms':>14} {'saved':>7}")
        for seq in (8192, 16384, 32768):
            on = spawn(workers.bench_ring, args.world, seq, 16, 128, 1, True)[0]
            off = spawn(workers.bench_ring, args.world, seq, 16, 128, 1, False)[0]
            rep(f"{seq:>7} {on['time_s'] * 1e3:>11.1f} {off['time_s'] * 1e3:>14.1f} "
                f"{1 - on['time_s'] / off['time_s']:>7.1%}")
        rep.save()
        return
    rep = Report("06_context_parallel.txt")
    rep("q, k, v: batch 1, 4 heads, head dim 8, fp32; causal; compared with full attention on one device")
    rep(f"{'C':>2} {'seq':>4} {'layout':>11} {'ring max err':>13}  score blocks per rank (in (s/2C)^2 units)  "
        f"max/mean")
    for C in (2, 4):
        seq = 16 * C
        for layout in ("contiguous", "zigzag"):
            rs = spawn(workers.cp_check, C, seq, 4, 8, 1, layout)
            units = [r["blocks"] * (4 if layout == "contiguous" else 1) for r in rs]
            rep(f"{C:>2} {seq:>4} {layout:>11} {max(r['ring_err'] for r in rs):>13.1e}  {str(units):<40} "
                f"{max(units) / (sum(units) / C):.2f}")
            if layout == "contiguous":
                ring = rs[0]["ring_ledger"]["ring_kv_send/fwd"]
                uly = rs[0]["ulysses_ledger"]
                rep(f"{'':>19} Ulysses: max err {max(r['ulysses_err'] for r in rs):.1e}, max grad err "
                    f"{max(r['ulysses_grad_err'] for r in rs):.1e}")
                rep(f"{'':>19} rank 0 sends, forward: ring {ring['count']} KV blocks = {ring['wire']} B; Ulysses "
                    f"{uly['all_to_all/fwd']['count']} all-to-alls = {uly['all_to_all/fwd']['wire']} B")
    rep("")
    rep("Contiguous shards: the last rank computes C blocks, the first one block, so the step waits on the last.")
    rep("Zigzag gives every rank chunk r and chunk 2C-1-r, and every rank computes 2C+1 sub-blocks.")
    rep("Ring traffic per rank is (C-1) KV blocks, which grows with s; Ulysses moves (C-1)/C of q, k, v and the output.")
    rep.save()


if __name__ == "__main__":
    main()
