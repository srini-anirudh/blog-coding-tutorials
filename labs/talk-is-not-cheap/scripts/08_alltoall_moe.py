"""Section 12: AllToAll moves half of AllReduce's bytes but has no locality, and MoE makes its messages unequal.

Numpy simulator and a synthetic router. No GPU, no real model.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from commcost import fabric  # noqa: E402
from commcost import schedules as S  # noqa: E402
from commcost import simulate as M  # noqa: E402
from commcost.report import Report, table  # noqa: E402

R = Report("08_alltoall_moe", note="numpy simulator and synthetic router, seed 0")
data = {}

R("1. Bytes per rank, as a fraction of n, on 4 nodes x 8 GPUs (P = 32). 'network' counts bytes that leave the node.")
nodes, g = 4, 8
p = nodes * g
n = p * g * 4
xs = [np.arange(n, dtype=float) + 1000 * r for r in range(p)]
out, a2a = M.run(S.alltoall_pairwise(p, n), xs, node_size=g)
blk = n // p
for i in range(p):  # check the transpose: slot j of rank i holds block i of rank j
    for j in range(p):
        assert np.array_equal(out[i]["y"][j * blk:(j + 1) * blk], xs[j][i * blk:(i + 1) * blk])
_, ring = M.run(S.ring(p, n), xs, node_size=g)
_, hier = M.run(S.hierarchical(nodes, g, n), xs, node_size=g)
rows = []
for name, st in [("AllToAll (pairwise)", a2a), ("AllReduce, flat ring", ring), ("AllReduce, hierarchical", hier)]:
    tot = st.max_sent / (n * 8)
    net = st.busiest_nic / (n * 8)
    rows.append([name, f"{tot:.3f}", f"{net:.3f}", f"{st.cross_node_bytes / sum(st.sent.values()):.0%}"])
    data[name] = {"sent_over_n": tot, "busiest_nic_over_n": net}
for line in table(rows, ["collective", "sent / n", "busiest NIC / n", "share of bytes leaving node"]):
    R(line)
R("AllToAll sends half of what AllReduce sends, but 3/4 of it must leave the node, and no reordering helps: every")
R("pair really exchanges. The hierarchical AllReduce puts 4x fewer bytes on its busiest NIC than AllToAll does.")

R()
R("2. MoE dispatch: 8 ranks x 8 experts, top-2, 4096 tokens per rank. The slowest receiver sets the AllToAll time.")
rows, moe = [], []
for skew in (0.0, 0.25, 0.5, 1.0):
    free = fabric.moe_dispatch(4096, 8, 8, 2, skew, np.random.default_rng(1))
    capped = fabric.moe_dispatch(4096, 8, 8, 2, skew, np.random.default_rng(1), capacity_factor=1.25)
    rows.append([skew, f"{free['max_over_mean']:.2f}", f"{capped['max_over_mean']:.2f}",
                 f"{capped['drop_fraction']:.1%}"])
    moe.append({"skew": skew, "max_over_mean": free["max_over_mean"], "capped_max_over_mean": capped["max_over_mean"],
                "drop_fraction": capped["drop_fraction"]})
for line in table(rows, ["router skew", "busiest / mean rank", "with capacity 1.25", "tokens dropped"]):
    R(line)
R("With equal messages the AllToAll cost model holds. As routing skews, the busiest rank receives more than its share")
R("and everyone waits for it. A capacity factor caps that ratio and pays for it in dropped tokens.")
data["moe"] = moe
R.save(data)
