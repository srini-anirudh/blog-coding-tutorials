"""Section 9: the distance to the data. Load latency at each rung of the GPU memory ladder, by pointer chasing.

One thread follows a random cycle through an array (one node per 256 bytes). Each address is the value of the previous load, so no two loads
overlap and the time per step is the load latency. The array size picks the rung: it fits in L1, fits in L2, or
spills to HBM. Then Little's Law turns the HBM latency into the bytes that must be in flight to reach peak bandwidth.

    python labs/why-gpus-are-built-for-deep-learning/scripts/02_memory_ladder.py
"""

import _path  # noqa: F401
import torch

from whygpu import kernels as K
from whygpu.hw import H100_SPEC, Report

SPACING = 64  # words between nodes (256 bytes), so no two nodes share a sector or a line


def build_cycle(n_bytes: int) -> torch.Tensor:
    """A single random cycle over n_bytes / 256 nodes (Sattolo order via a random permutation)."""
    words = n_bytes // 4
    m = words // SPACING
    perm = torch.randperm(m, device="cuda", dtype=torch.int64)
    perm = perm[perm != 0]
    order = torch.cat([torch.zeros(1, device="cuda", dtype=torch.int64), perm]) * SPACING
    nxt = torch.zeros(words, device="cuda", dtype=torch.int32)
    nxt[order] = torch.roll(order, -1).to(torch.int32)
    return nxt


def chase(nxt: torch.Tensor, load: str, steps: int, cached: bool) -> tuple[float, float]:
    """Cycles and ns per load. cached=True walks the whole cycle first, so every timed load hits the cache.
    cached=False flushes L2 first; the timed walk covers far fewer nodes than exist, so it never revisits one."""
    k = K.get("chase", LOAD=load)
    cyc = torch.zeros(1, dtype=torch.int64, device="cuda")
    sink = torch.zeros(1, dtype=torch.int32, device="cuda")
    nodes = nxt.numel() // SPACING
    k(grid=(1, 1, 1), block=(1, 1, 1), args=[nxt, nodes if cached else 1, cyc, sink])
    if not cached:
        flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
        flush.fill_(1)  # push nxt out of the 50 MB L2
        del flush
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    k(grid=(1, 1, 1), block=(1, 1, 1), args=[nxt, steps, cyc, sink])
    end.record()
    torch.cuda.synchronize()
    return cyc.item() / steps, start.elapsed_time(end) * 1e6 / (steps + 64)


def main() -> None:
    rep = Report("02_memory_ladder_h100.txt", "cuda")
    cases = [
        ("16 KB, plain load", 16 << 10, "*", 200_000, "L1"),
        ("16 KB, __ldcg (skip L1)", 16 << 10, "__ldcg", 200_000, "L2"),
        ("4 MB, __ldcg", 4 << 20, "__ldcg", 200_000, "L2"),
        ("32 MB, __ldcg", 32 << 20, "__ldcg", 200_000, "L2, both halves"),
        ("256 MB, __ldcg", 256 << 20, "__ldcg", 50_000, "HBM"),
        ("2 GB, __ldcg", 2 << 30, "__ldcg", 50_000, "HBM, more TLB misses"),
    ]
    rows, hbm_ns = [], None
    for label, size, load, steps, rung in cases:
        nxt = build_cycle(size)
        cycles, ns = chase(nxt, load, steps, cached=size <= 32 << 20)
        rows.append([label, rung, cycles, ns, cycles / ns])
        if size == 256 << 20:
            hbm_ns = ns
        del nxt
        torch.cuda.empty_cache()
    rep.table(["array", "rung", "cycles/load", "ns/load", "GHz (cyc/ns)"], rows)
    bw = H100_SPEC["hbm_tbps"] * 1e12
    inflight = hbm_ns * 1e-9 * bw
    sms = torch.cuda.get_device_properties().multi_processor_count
    rep(f"\nLittle's Law at {hbm_ns:.0f} ns and {H100_SPEC['hbm_tbps']} TB/s: {inflight / 1e6:.2f} MB in flight, "
        f"{inflight / sms / 1024:.1f} KB per SM, {inflight / sms / 16:,.0f} 16-byte loads per SM.")
    rep("ns/load comes from CUDA events and includes one launch (amortised over all steps).")
    rep.save()


if __name__ == "__main__":
    main()
