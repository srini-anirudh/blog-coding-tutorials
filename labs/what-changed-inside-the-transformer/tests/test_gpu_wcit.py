import importlib.util
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from organs.small import chunked_cross_entropy

pytestmark = pytest.mark.gpu
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _bench():
    path = Path(__file__).resolve().parents[1] / "benchmarks" / "gpu_bench.py"
    spec = importlib.util.spec_from_file_location("gpu_bench", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@needs_cuda
@pytest.mark.parametrize("nkv", [8, 2, 1])
def test_grouped_decode_matches_sdpa(nkv):
    q = torch.randn(2, 8, 1, 64, device="cuda")
    k, v = torch.randn(2, nkv, 300, 64, device="cuda"), torch.randn(2, nkv, 300, 64, device="cuda")
    ref = F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
    assert torch.allclose(_bench().grouped_decode(q, k, v), ref, atol=1e-4)


@needs_cuda
def test_chunked_ce_on_cuda():
    h = torch.randn(256, 32, device="cuda", requires_grad=True)
    w = torch.randn(1000, 32, device="cuda", requires_grad=True)
    y = torch.randint(0, 1000, (256,), device="cuda")
    assert chunked_cross_entropy(h, w, y, 64).item() == pytest.approx(F.cross_entropy(h @ w.T, y).item(), rel=1e-5)
