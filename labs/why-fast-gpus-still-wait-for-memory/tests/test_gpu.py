"""Triton kernels agree with PyTorch. Needs CUDA and Triton."""

import itertools

import pytest
import torch

pytestmark = pytest.mark.gpu
if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)
kernels = pytest.importorskip("rooflab.kernels")


@pytest.mark.parametrize("programs", [1, 7, 132])
def test_copy(programs):
    x = torch.randn(100_003, device="cuda")
    y = torch.empty_like(x)
    kernels.copy(x, y, programs=programs)
    assert torch.equal(x, y)


@pytest.mark.parametrize("iters", [0, 8, 64])
def test_fma(iters):
    x = torch.randn(5000, device="cuda")
    y = torch.empty_like(x)
    kernels.fma_sweep(x, y, iters)
    ref = x.clone()
    for _ in range(iters):
        ref = ref * 0.9999 + x
    torch.testing.assert_close(y, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("breaks", list(itertools.product([True, False], repeat=4)))
def test_chain_every_boundary_choice(breaks):
    torch.manual_seed(0)
    x, r, g = (torch.randn(64, 1000, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    w = torch.randn(1000, device="cuda", dtype=torch.bfloat16)
    h, y = kernels.chain(x, r, g, w, list(breaks))
    h_ref, y_ref = kernels.chain_reference(x, r, g, w)
    torch.testing.assert_close(h, h_ref)
    torch.testing.assert_close(y, y_ref, rtol=0.05, atol=0.05)
