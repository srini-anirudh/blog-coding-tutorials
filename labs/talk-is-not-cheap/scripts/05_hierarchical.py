"""Sections 7 and 13: a flat ring on a two-level machine, against the hierarchical AllReduce.

Model only: the numpy simulator runs the real schedules and prices each message with the link it crosses
(450 GB/s inside a node, 50 GB/s between nodes, per GPU per direction, as in the post's Figure 16).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from commcost import alphabeta as ab  # noqa: E402
from commcost import schedules as S  # noqa: E402
from commcost import simulate as M  # noqa: E402
from commcost.report import Report, table  # noqa: E402

R = Report("05_hierarchical", note="numpy simulator, model times")
NV, NET = 450e9, 50e9
A_NV, A_NET = 2e-6, 3e-6
N_BYTES = 1 << 30  # 1 GiB of gradient per GPU


def price(schedule, p, g, bw_fast, bw_slow, n_bytes=N_BYTES, elems=None):
    elems = elems or p * g * 4
    xs = [np.random.default_rng(r).integers(0, 9, elems).astype(float) for r in range(p)]
    out, st = M.run(schedule(elems), xs, link=M.two_level(g, A_NV, 1 / bw_fast, A_NET, 1 / bw_slow), node_size=g,
                    bytes_per_element=n_bytes / elems)
    want = np.sum(xs, axis=0)
    assert all(np.array_equal(o["x"], want) for o in out)
    return st


rows, data = [], []
for nodes, g in [(4, 4), (4, 8), (16, 8)]:
    p = nodes * g
    flat = price(lambda n: S.ring(p, n), p, g, NV, NET)
    hier = price(lambda n: S.hierarchical(nodes, g, n), p, g, NV, NET)
    factor = g * (p - 1) / p * nodes / (nodes - 1)
    closed = ab.t_hierarchical(N_BYTES, g, nodes, A_NV, 1 / NV, A_NET, 1 / NET)
    rows.append([f"{nodes} x {g}", f"{flat.time * 1e3:.1f}", f"{hier.time * 1e3:.1f}", f"{closed * 1e3:.1f}",
                 f"{flat.time / hier.time:.1f}x", f"{flat.busiest_nic / hier.busiest_nic:.2f}x", f"{factor:.2f}x",
                 f"{flat.cross_node_bytes / hier.cross_node_bytes:.2f}x"])
    data.append({"nodes": nodes, "g": g, "flat_s": flat.time, "hier_s": hier.time, "closed_s": closed,
                 "flat_net_bytes": flat.cross_node_bytes, "hier_net_bytes": hier.cross_node_bytes,
                 "flat_busiest_nic": flat.busiest_nic, "hier_busiest_nic": hier.busiest_nic})
R("1 GiB AllReduce. NVLink 450 GB/s, alpha 2 us. Network 50 GB/s, alpha 3 us. Ranks numbered node-major.")
for line in table(rows, ["nodes x GPUs", "flat ring ms", "hierarchical ms", "closed form ms", "speed-up",
                         "busiest NIC bytes cut", "G(P-1)/P N/(N-1)", "total fabric bytes cut"]):
    R(line)
R("The busiest NIC's bytes fall by the post's factor, about G, and that NIC sets the time. Total bytes on the fabric")
R("barely change: a flat ring pushes everything through the N NICs at node boundaries, while the hierarchical")
R("version spreads nearly the same total over all G x N NICs, one rail each. So 'G times less fabric traffic' is")
R("true per NIC, not for the fabric as a whole.")

R()
p, g = 16, 4
chunk = N_BYTES / p
slow_round = A_NET + chunk / NET
fast_busy = A_NV + chunk / NV
R(f"Flat ring on 4 x 4: every round waits for the network hop ({slow_round * 1e3:.2f} ms). An NVLink hop needs")
R(f"{fast_busy * 1e3:.2f} ms of it, so the 12 NVLink hops are busy {fast_busy / slow_round:.0%} of each round.")

R()
R("Nuance 12: the two factors are separate. Same 4 x 8 machine, but the in-node links made as slow as the network:")
same = price(lambda n: S.hierarchical(4, 8, n), 32, 8, NET, NET)
flat_same = price(lambda n: S.ring(32, n), 32, 8, NET, NET)
R(f"busiest NIC bytes still cut {flat_same.busiest_nic / same.busiest_nic:.2f}x; "
  f"time {flat_same.time * 1e3:.1f} ms flat vs {same.time * 1e3:.1f} ms hierarchical.")
R("The per-NIC traffic cut holds whatever the link speeds, as the post says, but on its own it buys almost no time:")
R("the in-node phases now move 2(G-1)/G n over slow links. The speed-up needs both factors at once.")
data.append({"equal_links_flat_s": flat_same.time, "equal_links_hier_s": same.time})
R.save(data)
