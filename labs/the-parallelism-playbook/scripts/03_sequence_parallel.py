"""Sequence parallelism on top of TP: each AllReduce becomes a ReduceScatter plus an AllGather, the regions
between them are split along the sequence, and the bytes on the wire stay the same. Also shows the part the
simple story leaves out: to keep the saving in the matmul's saved input, Megatron gathers it a second time."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ppbook import workers  # noqa: E402
from ppbook.dist_utils import Report, spawn  # noqa: E402

VARIANTS = [("TP", False, False), ("TP+SP", True, False), ("TP+SP, regather", True, True)]


def main():
    rep = Report("03_sequence_parallel.txt")
    hidden, heads, seq, batch = 256, 4, 32, 2
    rep(f"one block: hidden {hidden}, {heads} heads, seq {seq}, batch {batch}, fp32. Activation bytes = tensors autograd")
    rep("saves for backward on one rank (deduplicated). Wire = bytes one rank sends with ring algorithms.")
    for t in (2, 4):
        rep("")
        rep(f"T = {t}")
        rep(f"  {'variant':<16} {'out err':>8} {'grad err':>9} {'wire, layer path':>17} {'wire, LN grads':>15}"
            f" {'activation bytes':>17} {'vs 1 device':>12}")
        for name, sp, rg in VARIANTS:
            r = spawn(workers.tp_check, t, hidden, heads, seq, batch, sp, 0, rg)[0]
            layer = sum(v["wire"] for k, v in r["ledger"].items() if not k.endswith("grad_sync"))
            sync = sum(v["wire"] for k, v in r["ledger"].items() if k.endswith("grad_sync"))
            rep(f"  {name:<16} {r['out_err']:>8.1e} {r['grad_err']:>9.1e} {layer:>17} {sync:>15}"
                f" {r['activation_bytes']:>17} {r['activation_bytes'] / r['ref_activation_bytes']:>11.1%}")
            rep("     collectives: " + ", ".join(f"{k} x{v['count']}" for k, v in r["ledger"].items()))
        rep(f"  replicated region per layernorm input: s*b*h*4 = {seq * batch * hidden * 4} B; SP keeps 1/{t} of it")
    rep("")
    rep("Reading it: TP+SP sends the same layer-path bytes as TP (AllReduce = ReduceScatter + AllGather). Plain SP")
    rep("still saves the gathered full-sequence input of each column-parallel matmul, so part of the saving is lost.")
    rep("Saving only the shard and gathering again in backward recovers it, at the cost of one extra AllGather per")
    rep("column-parallel matmul in the backward pass. SP also adds a small AllReduce of layernorm and bias gradients.")
    rep.save()


if __name__ == "__main__":
    main()
