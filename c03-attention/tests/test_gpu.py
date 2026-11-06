"""FlashAttention backend. GPU-only: run with `pytest c03-attention -m gpu` on a CUDA machine."""

import pytest
import torch

from attnzoo.backends import attend

pytestmark = pytest.mark.gpu


def needs_cuda():
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")


@pytest.mark.parametrize("hkv", [8, 2])
@pytest.mark.parametrize("causal", [False, True])
def test_flash_matches_naive(hkv, causal):
    needs_cuda()
    q = torch.randn(2, 8, 512, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(2, hkv, 512, 64, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    ref = attend(q.float(), k.float(), v.float(), backend="naive", causal=causal)
    out = attend(q, k, v, backend="flash", causal=causal).float()
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)


def test_flash_window_and_decode():
    needs_cuda()
    pytest.importorskip("flash_attn")
    q = torch.randn(1, 8, 4, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 2, 300, 64, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    ref = attend(q.float(), k.float(), v.float(), backend="naive", causal=True, window_size=128)
    out = attend(q, k, v, backend="flash", causal=True, window_size=128).float()
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)


def test_flash_refuses_cpu():
    q = torch.randn(1, 1, 4, 8)
    with pytest.raises(RuntimeError):
        attend(q, q, q, backend="flash")
