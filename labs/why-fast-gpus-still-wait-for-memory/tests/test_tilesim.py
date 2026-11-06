"""The traffic counter agrees with the closed forms at both ends of the cache range."""

import pytest

from rooflab import model as M
from rooflab.tilesim import LRU, simulate, tile_order


@pytest.mark.parametrize("t", [32, 64, 128])
def test_no_cache_matches_tiled_formula(t):
    n = 512
    assert simulate(n, n, n, 2, t, t, 32, cache_bytes=0).hbm_bytes == M.gemm_bytes_tiled(n, n, n, 2, t, t)


def test_unbounded_cache_is_compulsory():
    m, n, k = 256, 384, 512
    r = simulate(m, n, k, 2, 64, 64, 32, cache_bytes=float("inf"), wave=7)
    assert r.hbm_bytes == M.gemm_bytes(m, n, k, 2)


def test_traffic_falls_with_cache_and_grouping_helps():
    # The wave must span more than one group column for grouping to matter, as on a real GPU.
    n, kw = 1024, dict(tm=64, tn=64, tk=32, wave=16)
    q0 = simulate(n, n, n, cache_bytes=0, **kw).hbm_bytes
    row = simulate(n, n, n, cache_bytes=2**18, order="row", **kw).hbm_bytes
    grp = simulate(n, n, n, cache_bytes=2**18, order="grouped", **kw).hbm_bytes
    assert grp < row < q0
    assert grp >= M.gemm_bytes(n, n, n, 2)


def test_tile_orders_are_permutations():
    for order in ("row", "grouped"):
        tiles = tile_order(6, 5, order, group=4)
        assert sorted(tiles) == [(i, j) for i in range(6) for j in range(5)]


def test_lru_evicts_oldest():
    c = LRU(2)
    assert not c.access("a", 1) and not c.access("b", 1)
    assert c.access("a", 1)  # a is now most recent
    c.access("c", 1)  # evicts b
    assert c.access("a", 1) and not c.access("b", 1)
