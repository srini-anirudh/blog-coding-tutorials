"""Collective algorithms written as schedules: lists of rounds, each a list of point-to-point transfers.

A schedule says nothing about how bytes move. `simulate.py` runs it on numpy arrays and prices it with the alpha-beta
model. `dist_run.py` runs the very same schedule with torch.distributed send/recv on real processes or GPUs.

Round semantics: every transfer in a round reads the sender's buffer as it was at the start of the round. Receivers
apply what they got ("add" or "copy") after the round ends. A buffer is a flat 1-D array of N elements.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Transfer:
    src: int
    dst: int
    lo: int  # element range [lo, hi) read from the sender's "x" buffer
    hi: int
    op: str  # "add" or "copy"
    dst_lo: int | None = None  # where it lands on the receiver; default the same offset
    dst_buf: str = "x"  # alltoall writes into a separate output buffer "y"

    @property
    def count(self) -> int:
        return self.hi - self.lo

    @property
    def at(self) -> int:
        return self.lo if self.dst_lo is None else self.dst_lo


Schedule = list[list[Transfer]]


def chunks(n: int, k: int) -> list[tuple[int, int]]:
    """Split [0, n) into k contiguous pieces whose sizes differ by at most one."""
    base, extra = divmod(n, k)
    out, lo = [], 0
    for i in range(k):
        hi = lo + base + (1 if i < extra else 0)
        out.append((lo, hi))
        lo = hi
    return out


def merge(*schedules: Schedule) -> Schedule:
    """Run several schedules side by side: round i of the result is round i of every input."""
    depth = max((len(s) for s in schedules), default=0)
    return [[t for s in schedules if i < len(s) for t in s[i]] for i in range(depth)]


# ---------------------------------------------------------------- AllReduce algorithms
def naive(p: int, n: int) -> Schedule:
    """Everyone sends to rank 0 one at a time, rank 0 sends the sum back one at a time: 2(p-1) rounds of n."""
    up = [[Transfer(r, 0, 0, n, "add")] for r in range(1, p)]
    down = [[Transfer(0, r, 0, n, "copy")] for r in range(1, p)]
    return up + down


def binomial_tree(p: int, n: int) -> Schedule:
    """Pairwise reduce to rank 0 in ceil(log2 p) rounds, then broadcast back the same way. Full vector every round."""
    levels = math.ceil(math.log2(p)) if p > 1 else 0
    up, down = [], []
    for k in range(levels):
        d = 1 << k
        up.append([Transfer(r, r - d, 0, n, "add") for r in range(p) if r % (2 * d) == d])
    for k in reversed(range(levels)):
        d = 1 << k
        down.append([Transfer(r, r + d, 0, n, "copy") for r in range(0, p, 2 * d) if r + d < p])
    return up + down


def ring_reduce_scatter(ranks: list[int], lo: int, hi: int) -> tuple[Schedule, dict[int, tuple[int, int]]]:
    """Ring reduce-scatter of [lo, hi) over `ranks`. Returns the rounds and which slice each rank owns afterwards."""
    p = len(ranks)
    ch = [(lo + a, lo + b) for a, b in chunks(hi - lo, p)]
    rounds = []
    for s in range(p - 1):
        rounds.append([Transfer(ranks[i], ranks[(i + 1) % p], *ch[(i - s) % p], "add") for i in range(p)])
    owned = {ranks[i]: ch[(i + 1) % p] for i in range(p)}
    return rounds, owned


def ring_all_gather(ranks: list[int], lo: int, hi: int) -> Schedule:
    """Ring all-gather of [lo, hi), assuming rank i already holds the finished chunk (i + 1) mod p."""
    p = len(ranks)
    ch = [(lo + a, lo + b) for a, b in chunks(hi - lo, p)]
    return [
        [Transfer(ranks[i], ranks[(i + 1) % p], *ch[(i - s + 1) % p], "copy") for i in range(p)] for s in range(p - 1)
    ]


def ring(p: int, n: int, ranks: list[int] | None = None, lo: int = 0, hi: int | None = None) -> Schedule:
    ranks = list(range(p)) if ranks is None else ranks
    hi = n if hi is None else hi
    rs, _ = ring_reduce_scatter(ranks, lo, hi)
    return rs + ring_all_gather(ranks, lo, hi)


def recursive_doubling(p: int, n: int) -> Schedule:
    """Exchange the whole vector with partner r XOR d and add. log2 p rounds, n bytes each. p a power of two."""
    _check_pow2(p)
    return [[Transfer(r, r ^ (1 << k), 0, n, "add") for r in range(p)] for k in range(int(math.log2(p)))]


def rabenseifner(p: int, n: int) -> Schedule:
    """Recursive-halving reduce-scatter, then recursive-doubling all-gather. Hits both optimality floors."""
    _check_pow2(p)
    rng = {r: (0, n) for r in range(p)}
    history, rounds = [], []
    d = p // 2
    while d >= 1:
        rnd, new = [], {}
        for r in range(p):
            lo, hi = rng[r]
            mid = (lo + hi) // 2
            keep, give = ((lo, mid), (mid, hi)) if r & d == 0 else ((mid, hi), (lo, mid))
            rnd.append(Transfer(r, r ^ d, *give, "add"))
            new[r] = keep
        history.append(dict(rng))
        rng = new
        rounds.append(rnd)
        d //= 2
    d = 1
    while d < p:
        rounds.append([Transfer(r, r ^ d, *rng[r], "copy") for r in range(p)])
        rng = history.pop()
        d *= 2
    return rounds


def hierarchical(nodes: int, g: int, n: int) -> Schedule:
    """Two-level AllReduce. Ranks are numbered node-major: rank = node * g + local.

    1. Ring reduce-scatter inside every node (fast links). Local rank l ends up owning slice c_l.
    2. Ring AllReduce of slice c_l across nodes among the ranks with the same local index (one rail each).
    3. Ring all-gather inside every node.
    """
    node_ranks = [[j * g + l for l in range(g)] for j in range(nodes)]
    rs_parts, owned = [], {}
    for ranks in node_ranks:
        rs, own = ring_reduce_scatter(ranks, 0, n)
        rs_parts.append(rs)
        owned.update(own)
    inter_parts = []
    for l in range(g):
        rail = [j * g + l for j in range(nodes)]
        lo, hi = owned[rail[0]]  # every node chunks the same way, so the whole rail owns the same slice
        if nodes > 1:
            inter_parts.append(ring(nodes, n, ranks=rail, lo=lo, hi=hi))
    ag_parts = [ring_all_gather(ranks, 0, n) for ranks in node_ranks]
    return merge(*rs_parts) + merge(*inter_parts) + merge(*ag_parts)


ALLREDUCE = {
    "naive": naive,
    "tree": binomial_tree,
    "ring": ring,
    "recursive_doubling": recursive_doubling,
    "rabenseifner": rabenseifner,
}


# ---------------------------------------------------------------- AllToAll
def alltoall_pairwise(p: int, n: int) -> Schedule:
    """Block j of rank i's x goes to slot i of rank j's y. Round s: rank i sends to rank i + s. Round 0 is local."""
    blk = chunks(n, p)
    rounds = []
    for s in range(p):
        rnd = []
        for i in range(p):
            j = (i + s) % p
            rnd.append(Transfer(i, j, *blk[j], "copy", dst_lo=blk[i][0], dst_buf="y"))
        rounds.append(rnd)
    return rounds


def _check_pow2(p: int) -> None:
    if p < 1 or p & (p - 1):
        raise ValueError(f"p must be a power of two, got {p}")
