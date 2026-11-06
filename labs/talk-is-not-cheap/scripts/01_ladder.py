"""Section 1, 3, 4, 5, 6 and 9 arithmetic: the bandwidth ladder, n_half, PCIe encoding, link counts.

Every input is a published spec quoted in the post. This script only checks the division.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from commcost import ladder  # noqa: E402
from commcost.alphabeta import n_half  # noqa: E402
from commcost.report import Report, table  # noqa: E402

R = Report("01_ladder", note="arithmetic only, no measurement")

R("The ladder: break-even arithmetic intensity against 989 TFLOP/s dense bf16")
rows = [
    [r["level"], f"{r['bandwidth_GBps']:,.0f}", f"{r['break_even_flop_per_byte']:,.0f}", f"{r['slower_than_hbm']:.2f}x"]
    for r in ladder.ladder_table()
]
for line in table(rows, ["level", "GB/s", "FLOP/byte", "slower than HBM"]):
    R(line)
bw = dict(ladder.LADDER)
R()
R(f"Cliff HBM -> NVLink: {bw['HBM3'] / bw['NVLink 4 (per direction)']:.1f}x")
R(f"Cliff NVLink -> network: {bw['NVLink 4 (per direction)'] / bw['Network (400 Gb/s port per GPU)']:.1f}x")
R(f"NVLink quoted bidirectional (900 GB/s) vs network per direction (50 GB/s): {900 / 50:.0f}x, the mixed convention")

R()
R("n_half = alpha / beta for the post's representative paths")
paths = [
    ("NVLink 4 via NVSwitch", 2e-6, 370e9),
    ("PCIe Gen5 peer-to-peer", 3e-6, 50e9),
    ("InfiniBand NDR", 3e-6, 45e9),
    ("EFA v2", 12e-6, 42e9),
]
rows = [[n, f"{a * 1e6:.0f} us", f"{b / 1e9:.0f} GB/s", f"{n_half(a, 1 / b) / 1e3:.0f} KB"] for n, a, b in paths]
for line in table(rows, ["path", "alpha", "1/beta", "n_half"]):
    R(line)

R()
lane = ladder.pcie_lane_bytes_per_s()
R(f"PCIe Gen5 lane after 128b/130b: {lane / 1e9:.3f} GB/s; x16: {ladder.pcie_link_bytes_per_s() / 1e9:.2f} GB/s")
R(f"PCIe Gen5 x16 before encoding: {32e9 * 16 / 8 / 1e9:.0f} GB/s")
gen4 = ladder.pcie_link_bytes_per_s(gt_per_s=16e9)
R(f"PCIe Gen4 x16: {gen4 / 1e9:.1f} GB/s, so a 400 Gb/s NIC behind it is capped at {gen4 / 50e9:.0%} of its rating")
R(f"NVLink to PCIe Gen5 x16 neighbour: {450e9 / ladder.pcie_link_bytes_per_s():.1f}x; "
  f"PCIe neighbour vs own HBM: {ladder.pcie_link_bytes_per_s() / 3.35e12:.1%}")

R()
bdp = ladder.bandwidth_delay_product(ladder.gbit_to_gbyte(400) * 1e9, 10e-6)
R(f"Bandwidth-delay product, 400 Gb/s x 10 us RTT: {bdp / 1e3:.0f} KB")
R(f"Propagation over 30 m at 5 ns/m: {ladder.propagation_s(30) * 1e9:.0f} ns")
for n in (4, 8, 16):
    total, per = ladder.full_mesh_links(n)
    R(f"Full mesh of {n} GPUs: {total} links, {per} per GPU")
R(f"Three-level fat tree from 64-port switches: {ladder.fat_tree_hosts(64):,} hosts")
R.save()
