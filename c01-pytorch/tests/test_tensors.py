import pytest
import torch

from scratchnet.broadcasting import broadcast_shape, expand_to
from scratchnet.matmul import bmm_loop, matmul_manual, mm_loops, mm_rows
from scratchnet.tensors import (
    contiguous_strides,
    describe,
    memory_bytes,
    permute_manual,
    shares_storage,
    storage_indices,
    transpose_manual,
    view_manual,
)


def test_memory_footprint():
    x = torch.randn(4, 8)
    assert memory_bytes(x) == 4 * 8 * 4
    assert memory_bytes(x.half()) == 4 * 8 * 2
    assert describe(x).nbytes == x.untyped_storage().nbytes()


def test_contiguous_strides_match_torch():
    for shape in [(4, 8), (2, 3, 4), (5,), (2, 1, 3, 1)]:
        assert contiguous_strides(shape) == torch.empty(shape).stride()


def test_storage_indices_match_as_strided():
    base = torch.arange(24)
    for t in [base.view(4, 6), base.view(4, 6).t(), base.view(2, 3, 4).permute(2, 0, 1), base.view(4, 6)[1:, ::2]]:
        assert torch.equal(storage_indices(t), t)  # base[i] == i, so values are storage indices


def test_transpose_and_permute_move_no_data():
    x = torch.randn(3, 4, 5)
    t = transpose_manual(x, 0, 2)
    assert torch.equal(t, x.transpose(0, 2)) and shares_storage(t, x) and not t.is_contiguous()
    p = permute_manual(x, (1, 2, 0))
    assert torch.equal(p, x.permute(1, 2, 0)) and p.stride() == x.permute(1, 2, 0).stride()


def test_view_vs_reshape_on_non_contiguous():
    x = torch.randn(4, 6).t()
    with pytest.raises(RuntimeError):
        x.view(24)
    with pytest.raises(RuntimeError):
        view_manual(x, (24,))
    r = x.reshape(24)  # reshape falls back to a copy
    assert not shares_storage(r, x)
    c = x.contiguous()
    assert torch.equal(view_manual(c, (24,)), c.view(24))


def test_slice_keeps_whole_storage_alive():
    big = torch.zeros(1000, 1000)
    row = big[0]
    assert describe(row).nbytes == 4000
    assert describe(row).storage_bytes == 4_000_000
    assert describe(row.clone()).storage_bytes == 4000


@pytest.mark.parametrize("a,b", [((2, 5, 8), (8,)), ((2, 4, 5, 8), (1, 4, 1, 8)), ((3, 1), (1, 4)), ((), (2, 3))])
def test_broadcast_shape_matches_torch(a, b):
    assert broadcast_shape(a, b) == tuple(torch.broadcast_shapes(a, b))
    x = torch.randn(b)
    assert torch.equal(expand_to(x, broadcast_shape(a, b)), x.expand(broadcast_shape(a, b)))


def test_broadcast_shape_rejects_mismatch():
    with pytest.raises(ValueError):
        broadcast_shape((2, 5, 8), (5,))


def test_expand_uses_stride_zero():
    bias = torch.randn(8)
    e = expand_to(bias, (2, 5, 8))
    assert e.stride() == (0, 0, 1) and shares_storage(e, bias)


def test_matmul_variants_agree():
    a, b = torch.randn(5, 7), torch.randn(7, 3)
    ref = a @ b
    torch.testing.assert_close(mm_loops(a, b), ref)
    torch.testing.assert_close(mm_rows(a, b), ref)
    torch.testing.assert_close(torch.mm(a, b), ref)
    ba, bb = torch.randn(4, 5, 7), torch.randn(4, 7, 3)
    torch.testing.assert_close(bmm_loop(ba, bb), torch.bmm(ba, bb))
    torch.testing.assert_close(torch.matmul(ba, bb), torch.bmm(ba, bb))


@pytest.mark.parametrize("a,b", [((5, 7), (7, 3)), ((4, 5, 7), (4, 7, 3)), ((4, 5, 7), (7, 3)), ((2, 1, 5, 7), (3, 7, 3))])
def test_matmul_manual_matches_torch(a, b):
    x, y = torch.randn(a), torch.randn(b)
    torch.testing.assert_close(matmul_manual(x, y), torch.matmul(x, y))
