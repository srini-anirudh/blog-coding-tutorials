"""The post's simplified models, written as plain functions.

Each function matches one of the post's figures or interactive labs. They count things (instructions,
passes, sectors, cycles, loads). They do not time anything. The scripts compare them with measurements.
"""

import math
import random

WARP = 32


# Little's Law (section 1) ---------------------------------------------------------------------------------------

def littles_law_concurrency(latency: float, throughput: float) -> float:
    """Operations that must be in flight: C = L x T."""
    return latency * throughput


# SIMD and the tail (Figure 05) -----------------------------------------------------------------------------------

def simd_loop(n: int, width: int) -> dict:
    """Vector instructions and wasted lanes for a contiguous loop of n elements on a `width`-lane register."""
    instr = math.ceil(n / width)
    issued = instr * width
    return {
        "instructions": instr,
        "lanes_issued": issued,
        "wasted_lanes": issued - n,
        "utilisation": n / issued if issued else 1.0,
        "speedup_vs_scalar": n / instr if instr else 1.0,
    }


# Divergence (Figures 08 and 09) ----------------------------------------------------------------------------------

def assign_paths(n_threads: int, paths: int, mode: str, seed: int = 0) -> list[int]:
    """Which path each thread takes. 'warp' gives every warp one path, 'tid' uses tid % paths."""
    if mode == "warp":
        return [(t // WARP) % paths for t in range(n_threads)]
    if mode == "tid":
        return [t % paths for t in range(n_threads)]
    if mode == "random":
        rng = random.Random(seed)
        return [rng.randrange(paths) for _ in range(n_threads)]
    raise ValueError(mode)


def divergence_passes(path_of_thread: list[int]) -> int:
    """Serialised passes: each warp runs every distinct path any of its lanes needs."""
    return sum(len(set(path_of_thread[w:w + WARP])) for w in range(0, len(path_of_thread), WARP))


def tripcount_cost(iters_of_thread: list[int]) -> int:
    """A loop costs each warp the largest trip count among its lanes."""
    return sum(max(iters_of_thread[w:w + WARP]) for w in range(0, len(iters_of_thread), WARP))


# Occupancy, compute capability 9.0 (Figure 11) -------------------------------------------------------------------

CC90 = {
    "warps_per_sm": 64,
    "blocks_per_sm": 32,
    "regs_per_sm": 65536,
    "reg_unit": 256,          # registers are allocated per warp in units of 256
    "smem_per_sm": 228 * 1024,
    "smem_per_block_max": 227 * 1024,
    "smem_unit": 128,
    "smem_reserved_per_block": 1024,  # the driver keeps 1 KB per block for itself (CUDA Guide, CC 8.x and 9.0)
    "processing_blocks": 4,           # each SM sub-partition owns a quarter of the register file
}


def occupancy(threads_per_block: int, regs_per_thread: int, smem_per_block: int,
              post: bool = False, arch: dict = CC90) -> dict:
    """Resident blocks and warps per SM, and which limit binds.

    post=True is the post's Figure 11 calculator. post=False adds two rules the driver applies:
    the 1 KB of shared memory it reserves per block, and register allocation per processing block
    (each of the 4 holds a quarter of the register file and whole warps only).
    """
    wpb = math.ceil(threads_per_block / WARP)
    regs_per_warp = math.ceil(regs_per_thread * WARP / arch["reg_unit"]) * arch["reg_unit"]
    if not regs_per_thread:
        reg_blocks = arch["blocks_per_sm"]
    elif post:
        reg_blocks = arch["regs_per_sm"] // (regs_per_warp * wpb)
    else:
        parts = arch["processing_blocks"]
        reg_blocks = parts * (arch["regs_per_sm"] // parts // regs_per_warp) // wpb
    lim = {"warp slots": arch["warps_per_sm"] // wpb, "register file": reg_blocks, "block cap": arch["blocks_per_sm"]}
    if smem_per_block > arch["smem_per_block_max"]:
        lim["shared memory"] = 0
    else:
        need = smem_per_block + (0 if post else arch["smem_reserved_per_block"])
        need = math.ceil(need / arch["smem_unit"]) * arch["smem_unit"]
        lim["shared memory"] = arch["smem_per_sm"] // need if need else arch["blocks_per_sm"]
    blocks = min(lim.values())
    warps = min(arch["warps_per_sm"], blocks * wpb)
    return {
        "blocks": blocks,
        "warps": warps,
        "occupancy": warps / arch["warps_per_sm"],
        "binding": [k for k, v in lim.items() if v == blocks],
    }


# Coalescing (Figure 13) ------------------------------------------------------------------------------------------

SECTOR = 32


def warp_sectors(stride: int, elem_bytes: int = 4, misalign_bytes: int = 0,
                 indices: list[int] | None = None) -> dict:
    """Distinct 32-byte sectors one warp-wide load touches. Thread t reads element t*stride (or indices[t])."""
    idx = indices if indices is not None else [t * stride for t in range(WARP)]
    sectors, used = set(), set()
    for i in idx:
        start = misalign_bytes + i * elem_bytes
        for b in range(start, start + elem_bytes):
            sectors.add(b // SECTOR)
            used.add(b)
    moved = len(sectors) * SECTOR
    return {"sectors": len(sectors), "bytes_moved": moved, "bytes_used": len(used),
            "efficiency": len(used) / moved}


# Shared memory banks (Figure 14) ---------------------------------------------------------------------------------

BANKS = 32


def bank_conflict_degree(word_addresses: list[int]) -> int:
    """Cycles for one warp access: the most distinct words any one bank must serve. Same word broadcasts."""
    per_bank: dict[int, set] = {}
    for a in word_addresses:
        per_bank.setdefault(a % BANKS, set()).add(a)
    return max(len(s) for s in per_bank.values())


def tile_access(pattern: str, pad: int = 0, stride: int = 1, col: int = 0) -> list[int]:
    """Word addresses a warp reads from a 32 x (32 + pad) float tile."""
    width = 32 + pad
    if pattern == "row":
        return [col * width + t for t in range(WARP)]
    if pattern == "column":
        return [t * width + col for t in range(WARP)]
    if pattern == "strided":
        return [t * stride for t in range(WARP)]
    if pattern == "broadcast":
        return [col] * WARP
    raise ValueError(pattern)


# Tiling (Figure 15) ----------------------------------------------------------------------------------------------

def counted_matmul(a, b, tile: int | None):
    """C = A @ B in Python, counting element loads from 'slow memory'.

    tile=None is the naive triple loop: every product loads both operands.
    tile=T loads each T x T tile of A and B once per tile step and reuses it from 'fast memory'.
    Returns (C, loads).
    """
    n = len(a)
    c = [[0.0] * n for _ in range(n)]
    loads = 0
    if tile is None:
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    loads += 2
                    c[i][j] += a[i][k] * b[k][j]
        return c, loads
    t = tile
    assert n % t == 0
    for i0 in range(0, n, t):
        for j0 in range(0, n, t):
            for k0 in range(0, n, t):
                ta = [row[k0:k0 + t] for row in a[i0:i0 + t]]  # one tile load each
                tb = [row[j0:j0 + t] for row in b[k0:k0 + t]]
                loads += 2 * t * t
                for i in range(t):
                    for j in range(t):
                        c[i0 + i][j0 + j] += sum(ta[i][k] * tb[k][j] for k in range(t))
    return c, loads


def matmul_traffic(n: int, tile: int | None) -> int:
    """Closed form of counted_matmul's load count: 2 n^3 / T."""
    return 2 * n ** 3 // (tile or 1)


# Arithmetic intensity (Figure 17) --------------------------------------------------------------------------------

def ridge_point(peak_flops: float, bandwidth: float) -> float:
    """FLOP per byte needed to keep the arithmetic units busy."""
    return peak_flops / bandwidth


# FLOPs per element are hand counts chosen to match the intensities in the post's Figure 17 (fp16, one read and
# one write per element): GELU 10 -> 2.5 FLOP/byte, LayerNorm 8 -> 2.0, softmax 5 -> 1.25, add 1 -> 0.17.
FLOPS_PER_ELEMENT = {"add": 1, "gelu": 10, "layernorm": 8, "softmax": 5}


def op_cost(op: str, numel: int, elem_bytes: int = 2, n: int = 0) -> tuple[float, float]:
    """(FLOPs, minimum bytes) for an op, reading inputs and writing outputs once."""
    if op == "add":
        return numel, 3 * numel * elem_bytes
    if op in ("gelu", "layernorm", "softmax"):
        return FLOPS_PER_ELEMENT[op] * numel, 2 * numel * elem_bytes
    if op == "matmul":  # n x n x n
        return 2 * n ** 3, 3 * n * n * elem_bytes
    raise ValueError(op)
