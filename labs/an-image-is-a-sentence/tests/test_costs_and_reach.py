import pytest
import torch

from vitfamily import flops as fl
from vitfamily import reach as rc
from vitfamily.attention import MHSA
from vitfamily.blocks import DividedSTBlock, PVTBlock, ViTBlock
from vitfamily.c03 import WindowAttention


def meta_flops(build, *shape, extra=()):
    with torch.device("meta"):
        mod, x = build(), torch.empty(*shape)
    return fl.measure_flops(lambda: mod(x, *extra))


@pytest.mark.parametrize("n,d", [(197, 768), (50, 64), (4097, 768)])
def test_vit_block_flops_match_formula(n, d):
    got = meta_flops(lambda: ViTBlock(d, 4), 1, n, d)
    ref = fl.vit_block_flops(n, d)
    assert got["linear"] == ref["linear"] and got["attention"] == ref["attention"]


def test_crossover_at_six_d():
    n = fl.crossover_tokens(64)
    ref = fl.vit_block_flops(n, 64)
    assert ref["linear"] == ref["attention"]


def test_swin_stage1_table():
    glob = meta_flops(lambda: MHSA(96, 3), 1, 3136, 96)
    win = meta_flops(lambda: WindowAttention(96, 3, 7, 0), 1, 56, 56, 96)
    assert glob["linear"] // 2 == fl.swin_msa_macs(56, 56, 96)["linear"] == 115_605_504
    assert glob["attention"] // 2 == fl.swin_msa_macs(56, 56, 96)["attention"]
    assert win["attention"] // 2 == fl.swin_wmsa_macs(56, 56, 96, 7)["attention"] == 29_503_488
    assert round(glob["total"] / win["total"], 1) == 13.8


def test_sra_block_flops_op_by_op():
    got = meta_flops(lambda: PVTBlock(64, 1, 8), 1, 3136, 64, extra=((56, 56),))
    ref = fl.sra_block_flops(3136, 64, 8)
    assert got["attention"] == ref["attention"]
    assert got["linear"] == ref["linear"] and got["conv"] == ref["conv"]


def test_divided_block_flops():
    got = meta_flops(lambda: DividedSTBlock(32, 2), 1, 4, 49, 32)
    ref = fl.divided_block_flops(4, 49, 32)
    assert got["attention"] == ref["attention"] and got["linear"] == ref["linear"]


def test_st_pairs_ratios():
    p8, p32 = fl.st_pairs(8, 196), fl.st_pairs(32, 196)
    assert round(p8["joint"] / p8["divided"], 1) == 7.7
    assert round(p32["joint"] / p32["divided"], 1) == 27.5


def test_unshifted_windows_never_cover():
    side, m = 14, 7
    assert rc.blocks_to_cover([rc.window_block(side, m)] * 6, 0) is None


def test_shifted_windows_cover_and_mask_blocks_the_seam():
    side, m = 14, 7
    w0, w1 = rc.window_block(side, m), rc.window_block(side, m, 3)
    assert rc.blocks_to_cover([w0, w1] * 4, 0) is not None
    masked = rc.reach_over_depth([w0, w1], 0)[-1]
    torus = rc.reach_over_depth([w0, rc.window_block(side, m, 3, masked=False)], 0)[-1]
    assert not masked[-1] and torus[-1]


def test_window_block_rows_have_m_squared_keys_without_shift():
    a = rc.window_block(14, 7)
    assert (a.sum(1) == 49).all() and torch.equal(a, a.T)


def test_axial_covers_in_two_and_global_in_one():
    assert rc.blocks_to_cover(rc.axial_blocks(8), 0) == 2
    assert rc.blocks_to_cover([rc.global_block(8)], 5) == 1


def test_space_time_keys_per_query():
    t, side = 4, 3
    n = side * side
    assert rc.keys_per_query(rc.st_passes("S", t, side)) == n
    assert rc.keys_per_query(rc.st_passes("ST", t, side)) == t * n
    assert rc.keys_per_query(rc.st_passes("T+S", t, side)) == t + n
    assert rc.keys_per_query(rc.st_passes("T+W+H", t, side)) == t + 2 * side
    assert rc.reach_after_block(rc.st_passes("T+S", t, side)) == t * n
