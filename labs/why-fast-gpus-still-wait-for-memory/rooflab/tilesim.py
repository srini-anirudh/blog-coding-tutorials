"""Count the HBM bytes a tiled GEMM moves, with an LRU cache standing in for L2.

The output is cut into Tm x Tn tiles. Each tile walks K in steps of Tk and needs one A block (Tm x Tk) and one
B block (Tk x Tn) per step. A block that is still in the cache costs nothing at HBM. Tiles run in waves of
`wave` tiles that advance through K together, a crude picture of many SMs sharing one L2.

With a zero-byte cache this reproduces the post's Q_tiled formula. With an unbounded cache it reproduces
Q_compulsory. In between, the answer depends on the order in which tiles are scheduled.
"""

from collections import OrderedDict
from dataclasses import dataclass


@dataclass
class TileTraffic:
    hbm_bytes: int
    block_requests: int
    hits: int

    @property
    def hit_rate(self) -> float:
        return self.hits / max(1, self.block_requests)


def tile_order(mt: int, nt: int, order: str, group: int = 8) -> list[tuple[int, int]]:
    """'row': row-major over tiles. 'grouped': `group` tile rows at a time, column by column inside the group
    (the swizzle Triton's matmul tutorial calls GROUP_SIZE_M)."""
    if order == "row":
        return [(i, j) for i in range(mt) for j in range(nt)]
    if order == "grouped":
        out = []
        for g0 in range(0, mt, group):
            rows = range(g0, min(g0 + group, mt))
            out += [(i, j) for j in range(nt) for i in rows]
        return out
    raise ValueError(order)


class LRU:
    def __init__(self, capacity_bytes: float):
        self.cap = capacity_bytes
        self.used = 0
        self.d: OrderedDict = OrderedDict()

    def access(self, key, nbytes: int) -> bool:
        """Return True on a hit. On a miss, insert and evict the least recently used blocks."""
        if key in self.d:
            self.d.move_to_end(key)
            return True
        if nbytes <= self.cap:
            self.d[key] = nbytes
            self.used += nbytes
            while self.used > self.cap:
                _, b = self.d.popitem(last=False)
                self.used -= b
        return False


def simulate(m: int, n: int, k: int, s: int = 2, tm: int = 128, tn: int = 128, tk: int = 64,
             cache_bytes: float = 0, order: str = "row", group: int = 8, wave: int = 1) -> TileTraffic:
    assert m % tm == 0 and n % tn == 0 and k % tk == 0
    mt, nt, kt = m // tm, n // tn, k // tk
    a_blk, b_blk = s * tm * tk, s * tk * tn
    cache = LRU(cache_bytes)
    tiles = tile_order(mt, nt, order, group)
    hbm = s * m * n  # every output element written once
    req = hits = 0
    for w0 in range(0, len(tiles), wave):
        batch = tiles[w0:w0 + wave]
        for kk in range(kt):
            for i, j in batch:
                for key, nb in ((("A", i, kk), a_blk), (("B", kk, j), b_blk)):
                    req += 1
                    if cache.access(key, nb):
                        hits += 1
                    else:
                        hbm += nb
    return TileTraffic(hbm, req, hits)
