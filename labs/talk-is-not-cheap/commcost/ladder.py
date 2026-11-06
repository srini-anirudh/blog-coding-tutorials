"""The bandwidth ladder of an H100 SXM node, and the unit arithmetic the post leans on.

Every constant here is copied from the post, which takes them from vendor documents. Nothing in this file is measured.
The functions only do the division, so a reader can check the derived columns.
"""

from __future__ import annotations

DENSE_BF16_FLOPS = 989e12  # H100 SXM dense bf16 tensor-core peak, FLOP/s

# Per-GPU bandwidth at each level, bytes per second, one direction.
LADDER = [
    ("Shared memory / L1", 29.7e12),
    ("L2 cache", 5.5e12),
    ("HBM3", 3.35e12),
    ("NVLink 4 (per direction)", 450e9),
    ("PCIe Gen5 x16", 63e9),
    ("Network (400 Gb/s port per GPU)", 50e9),
]


def break_even_intensity(bandwidth: float, flops: float = DENSE_BF16_FLOPS) -> float:
    """FLOPs you must do per byte fetched from a level to keep the tensor cores busy."""
    return flops / bandwidth


def ladder_table() -> list[dict]:
    hbm = dict(LADDER)["HBM3"]
    return [
        {
            "level": name,
            "bandwidth_GBps": bw / 1e9,
            "break_even_flop_per_byte": break_even_intensity(bw),
            "slower_than_hbm": hbm / bw,
        }
        for name, bw in LADDER
    ]


def pcie_lane_bytes_per_s(gt_per_s: float = 32e9, payload_bits: int = 128, line_bits: int = 130) -> float:
    """One lane, one direction, after line encoding (128b/130b through Gen5)."""
    return gt_per_s * payload_bits / line_bits / 8


def pcie_link_bytes_per_s(lanes: int = 16, **kw) -> float:
    return lanes * pcie_lane_bytes_per_s(**kw)


def gbit_to_gbyte(gbit_per_s: float) -> float:
    return gbit_per_s / 8


def bandwidth_delay_product(link_bytes_per_s: float, rtt_s: float) -> float:
    """Bytes that must be in flight at once to keep a link busy."""
    return link_bytes_per_s * rtt_s


def propagation_s(metres: float, ns_per_m: float = 5.0) -> float:
    return metres * ns_per_m * 1e-9


def full_mesh_links(n: int) -> tuple[int, int]:
    """Direct wiring of n GPUs: total links and links per GPU."""
    return n * (n - 1) // 2, n - 1


def fat_tree_hosts(radix: int) -> int:
    """Hosts in a three-level folded Clos built from radix-k switches (Al-Fares et al.)."""
    return radix**3 // 4
