import random

import pytest

from whygpu import models as M


def test_littles_law():
    assert M.littles_law_concurrency(4, 16896) == 67584


def test_simd_tail_matches_post_example():
    # The post: length 37 at 16 lanes, the last iteration runs at 5/16.
    m = M.simd_loop(37, 16)
    assert m["instructions"] == 3 and m["wasted_lanes"] == 11
    assert 37 - 2 * 16 == 5
    assert m["utilisation"] == pytest.approx(37 / 48)
    assert M.simd_loop(1024, 8)["speedup_vs_scalar"] == 8


@pytest.mark.parametrize("p", [1, 2, 4, 8])
def test_divergence_warp_aligned_is_free_and_tid_mod_costs_p(p):
    warps = 256 // 32
    assert M.divergence_passes(M.assign_paths(256, p, "warp")) == warps
    assert M.divergence_passes(M.assign_paths(256, p, "tid")) == warps * p


def test_tripcount_costs_the_warp_maximum():
    iters = [1] * 31 + [100] + [5] * 32
    assert M.tripcount_cost(iters) == 105


def test_occupancy_post_examples():
    assert M.occupancy(256, 32, 0)["occupancy"] == 1.0
    assert M.occupancy(256, 64, 0)["occupancy"] == 0.5
    assert M.occupancy(256, 128, 0)["blocks"] == 2
    assert "warp slots" in M.occupancy(1024, 32, 0)["binding"]
    assert M.occupancy(32, 16, 0)["blocks"] == 32  # block cap binds
    assert M.occupancy(256, 32, 228 * 1024)["blocks"] == 0  # over the per-block cap


def test_occupancy_reserved_shared_memory_changes_the_answer():
    # 57 KB: 228 // 57 = 4 blocks without the 1 KB reserve, 228 // 58 = 3 with it.
    assert M.occupancy(128, 32, 57 * 1024, post=True)["blocks"] == 4
    assert M.occupancy(128, 32, 57 * 1024)["blocks"] == 3


def test_occupancy_registers_are_split_across_processing_blocks():
    # 102 regs -> 3,328 per warp. Whole SM: 19 warps fit, so 9 two-warp blocks.
    # Per processing block: 16,384 // 3,328 = 4 warps, 16 in all, so 8 blocks (what the H100 driver reports).
    assert M.occupancy(64, 102, 0, post=True)["blocks"] == 9
    assert M.occupancy(64, 102, 0)["blocks"] == 8


def test_sectors():
    assert M.warp_sectors(1)["sectors"] == 4
    assert M.warp_sectors(32)["bytes_moved"] == 1024
    assert M.warp_sectors(8)["efficiency"] == 0.125
    assert M.warp_sectors(1, misalign_bytes=4)["sectors"] == 5
    assert M.warp_sectors(1, elem_bytes=2)["sectors"] == 2
    assert M.warp_sectors(1, elem_bytes=16)["sectors"] == 16
    assert M.warp_sectors(0, indices=[0] * 32)["sectors"] == 1


def test_bank_conflicts():
    assert M.bank_conflict_degree(M.tile_access("row")) == 1
    assert M.bank_conflict_degree(M.tile_access("column")) == 32
    assert M.bank_conflict_degree(M.tile_access("column", pad=1)) == 1
    assert M.bank_conflict_degree(M.tile_access("broadcast")) == 1
    for s, d in [(2, 2), (4, 4), (16, 16), (3, 1), (5, 1)]:
        assert M.bank_conflict_degree(M.tile_access("strided", stride=s)) == d


def test_counted_matmul_reuse_is_tile_size():
    rng = random.Random(1)
    n = 8
    a = [[rng.random() for _ in range(n)] for _ in range(n)]
    b = [[rng.random() for _ in range(n)] for _ in range(n)]
    ref, naive = M.counted_matmul(a, b, None)
    exact = [[sum(a[i][k] * b[k][j] for k in range(n)) for j in range(n)] for i in range(n)]
    for row, erow in zip(ref, exact):
        assert row == pytest.approx(erow)
    for t in (2, 4, 8):
        c, loads = M.counted_matmul(a, b, t)
        assert loads == M.matmul_traffic(n, t) == naive // t
        for row, erow in zip(c, ref):
            assert row == pytest.approx(erow)


def test_intensities_match_post_figure_17():
    ai = {op: (lambda f, b: f / b)(*M.op_cost(op, 1000, 2)) for op in ("add", "gelu", "layernorm", "softmax")}
    assert ai["add"] == pytest.approx(0.1667, abs=1e-3)
    assert ai["gelu"] == 2.5 and ai["layernorm"] == 2.0 and ai["softmax"] == 1.25
    assert round(M.ridge_point(989e12, 3.35e12)) == 295
    f, b = M.op_cost("matmul", 0, 2, n=8192)
    assert f / b > 2000
