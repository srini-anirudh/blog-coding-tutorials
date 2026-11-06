"""Three numbers into a verdict.

Measure useful FLOPs, bytes at the DRAM boundary and time. Then
    intensity = FLOPs / bytes,   achieved = FLOPs / time,   ceiling = min(peak, bandwidth * intensity)
and the kernel is on the sloped roof (bandwidth bound), on the flat roof (compute bound), or well below both.
"""

from __future__ import annotations

from dataclasses import dataclass

# Vendor numbers for an H100 SXM5 80 GB (the "NVIDIA H100 80GB HBM3"), dense, sparsity claim removed.
# They are datasheet values, not measurements. Use them as the denominator, as MFU does.
H100_SXM = {
    "bf16_flops": 989e12,
    "tf32_flops": 494e12,
    "fp32_flops": 67e12,  # non-tensor-core FP32
    "hbm_bytes_per_s": 3.35e12,
}


@dataclass
class Verdict:
    flops: float
    bytes: float
    seconds: float
    intensity: float        # FLOP / byte
    achieved: float         # FLOP / s
    ceiling: float          # FLOP / s the roofline allows at this intensity
    fraction_of_roof: float
    ridge: float            # intensity where the two roofs meet
    regime: str             # "bandwidth bound", "compute bound" or "below both roofs"
    bandwidth: float        # achieved bytes / s

    def row(self) -> str:
        return (f"I={self.intensity:8.2f} FLOP/B  {self.achieved / 1e12:8.2f} TFLOP/s  {self.bandwidth / 1e9:8.0f} GB/s  "
                f"{100 * self.fraction_of_roof:5.1f}% of roof  -> {self.regime}")


def verdict(flops: float, bytes_moved: float, seconds: float, peak_flops: float, peak_bw: float,
            near: float = 0.5) -> Verdict:
    """`near` is how close to a roof counts as being on it. The post's third case, far below both roofs,
    is the common one: it means latency, occupancy or launch overhead, and neither fewer FLOPs nor fewer
    bytes will help."""
    intensity = flops / bytes_moved
    achieved = flops / seconds
    bw_roof = peak_bw * intensity
    ceiling = min(peak_flops, bw_roof)
    frac = achieved / ceiling
    ridge = peak_flops / peak_bw
    if frac < near:
        regime = "below both roofs"
    elif bw_roof < peak_flops:
        regime = "bandwidth bound"
    else:
        regime = "compute bound"
    return Verdict(flops, bytes_moved, seconds, intensity, achieved, ceiling, frac, ridge, regime,
                   bytes_moved / seconds)


# ---- analytic counts. Analytic FLOPs with analytic bytes, never mixed with measured bytes. ------------------

def matmul_counts(m: int, n: int, k: int, elem_bytes: int) -> tuple[float, float]:
    """C[m,n] = A[m,k] @ B[k,n]: 2mnk FLOPs, and at best each operand crosses DRAM once."""
    return 2.0 * m * n * k, float(elem_bytes * (m * k + k * n + m * n))


def elementwise_counts(numel: int, n_inputs: int, n_outputs: int, flops_per_elem: float,
                       elem_bytes: int) -> tuple[float, float]:
    return flops_per_elem * numel, float(elem_bytes * numel * (n_inputs + n_outputs))


def time_saved(share_of_step: float, part_speedup: float) -> float:
    """The time budget a roofline point lacks: the fraction of the step saved by making one part
    `part_speedup` times faster. Tripling a 4% part saves 2.7%; halving an 18% part saves 9%."""
    return share_of_step * (1.0 - 1.0 / part_speedup)
