"""Tensor parallelism: a Megatron block split over T ranks gives the single-device result, issues exactly
2 AllReduces forward and 2 backward per layer, and its message is activation-sized, not parameter-sized."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ppbook import workers  # noqa: E402
from ppbook.dist_utils import Report, spawn  # noqa: E402


def fmt(ledger):
    return ", ".join(f"{k} x{v['count']} ({v['wire']} B sent)" for k, v in ledger.items())


def main():
    rep = Report("02_tensor_parallel.txt")
    rep("1. Column-parallel then row-parallel, against one device (hidden 64, 8 heads, seq 16, batch 2, fp32)")
    rep(f"{'T':>3} {'max |y - y_ref|':>16} {'max |dx - dx_ref|':>18} {'max grad err':>13}  collectives per layer")
    for t in (2, 4, 8):
        r = spawn(workers.tp_check, t, 64, 8, 16, 2)[0]
        rep(f"{t:>3} {r['out_err']:>16.2e} {r['dx_err']:>18.2e} {r['grad_err']:>13.2e}  {fmt(r['ledger'])}")
    rep("")
    rep("2. The message is the activation b*s*h, so it grows with the step; DDP's gradient AllReduce does not")
    rep(f"{'batch':>5} {'seq':>4} {'TP bytes per AllReduce':>23} {'b*s*h*4':>8} {'block params (bytes)':>21}")
    for b, s in ((2, 16), (8, 16), (8, 64)):
        r = spawn(workers.tp_check, 2, 64, 8, s, b)[0]
        per_call = r["ledger"]["all_reduce/fwd"]["bytes"] // r["ledger"]["all_reduce/fwd"]["count"]
        rep(f"{b:>5} {s:>4} {per_call:>23} {r['act_msg_bytes']:>8} {r['param_bytes']:>21}")
    rep("")
    rep("3. Why column first. Row-parallel first leaves partial sums before GELU, and GELU does not distribute over +")
    r = spawn(workers.row_first_check, 2)[0]
    rep(f"   GELU applied to partial sums, then summed: max error vs GELU of the full sum = {r['gelu_of_partials_err']:.3f}")
    rep(f"   column then row: error {r['col_then_row_err']:.1e}; forward collectives: {fmt(r['col_then_row_fwd'])}")
    rep(f"   row then column: error {r['row_then_col_err']:.1e}; forward collectives: {fmt(r['row_then_col_fwd'])}")
    rep("   Both are correct. Row-first needs two collectives in the MLP forward pass and its AllReduce carries the")
    rep("   4h-wide hidden activation, four times the bytes of column-first's single AllReduce.")
    rep.save()


if __name__ == "__main__":
    main()
