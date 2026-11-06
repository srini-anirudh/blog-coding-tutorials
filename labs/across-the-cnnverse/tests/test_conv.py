import pytest
import torch
import torch.nn.functional as F

from cnnverse import conv


def test_loops_match_torch():
    g = torch.Generator().manual_seed(0)
    x, w, b = torch.randn(2, 6, 6, generator=g), torch.randn(3, 2, 3, 3, generator=g), torch.randn(3, generator=g)
    for s, p in [(1, 0), (2, 1), (1, 1)]:
        assert torch.allclose(conv.conv2d_loops(x, w, b, s, p), F.conv2d(x[None], w, b, s, p)[0], atol=1e-5)


@pytest.mark.parametrize("k,s,p,d,g", [(3, 1, 1, 1, 1), (3, 2, 1, 1, 1), (5, 1, 2, 1, 1), (3, 1, 2, 2, 1),
                                       (3, 1, 1, 1, 4), (3, 2, 1, 1, 8), (1, 1, 0, 1, 1), (7, 4, 3, 1, 1)])
def test_im2col_matches_torch(k, s, p, d, g):
    gen = torch.Generator().manual_seed(1)
    x, w = torch.randn(2, 8, 13, 13, generator=gen), torch.randn(8, 8 // g, k, k, generator=gen)
    b = torch.randn(8, generator=gen)
    y = conv.conv2d_im2col(x, w, b, s, p, d, g)
    assert torch.allclose(y, F.conv2d(x, w, b, s, p, d, g), atol=1e-4)
    assert y.shape[-1] == conv.out_size(13, k, s, p, d)


def test_transposed_scatter_matches_torch():
    gen = torch.Generator().manual_seed(2)
    x, w = torch.randn(2, 3, 5, 5, generator=gen), torch.randn(3, 4, 3, 3, generator=gen)
    for s, p in [(2, 0), (2, 1), (1, 1), (3, 0)]:
        ref = F.conv_transpose2d(x, w, stride=s, padding=p)
        assert torch.allclose(conv.conv_transpose2d_scatter(x, w, s, p), ref, atol=1e-5)


def test_family_table_matches_post():
    rows = {n: (p, m) for n, p, m in conv.family_table()}
    assert rows["Standard 3x3"] == (147_456, 462_422_016)
    assert rows["1x1 (pointwise)"][0] == 16_384 and round(rows["1x1 (pointwise)"][1] / 1e6, 1) == 51.4
    assert rows["Depthwise 3x3"][0] == 1_152 and round(rows["Depthwise 3x3"][1] / 1e6, 1) == 3.6
    assert rows["Depthwise separable"][0] == 17_536 and round(rows["Depthwise separable"][1] / 1e6, 1) == 55.0
    assert rows["Grouped 3x3, g=32"][0] == 4_608 and round(rows["Grouped 3x3, g=32"][1] / 1e6, 1) == 14.5
    assert round(rows["Transposed 3x3, s=2"][1] / 1e6, 1) == 1849.7
    assert round(1 / conv.separable_ratio(3, 128), 1) == 8.4


def test_geometry_claims():
    assert conv.unpadded_shrink(224) == 192
    stacked, single, k = conv.stacked_vs_single(10)
    assert (stacked, single, k) == (27 * 100, 49 * 100, 7)
    assert conv.conv_macs(3, 64, 64, 16 * 56 * 56, kt=3) == 48 * conv.conv_macs(3, 64, 64, 56 * 56)


def test_kernel_sums_set_dc_response():
    flat = torch.ones(1, 1, 5, 5)
    for name, k in conv.KERNELS.items():
        y = F.conv2d(flat, k[None, None])
        assert torch.allclose(y, torch.full_like(y, float(k.sum())), atol=1e-6), name
