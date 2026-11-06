"""Checks that need a CUDA GPU: kernels run and give correct or sensible answers."""

import pytest
import torch

from whygpu import kernels as K
from whygpu.models import occupancy

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")]


def test_fma_latency_is_a_few_cycles():
    out = torch.zeros(32, device="cuda")
    cyc = torch.zeros(32, dtype=torch.int64, device="cuda")
    K.get("fma_chain")(grid=(1, 1, 1), block=(1, 1, 1), args=[out, cyc, 100_000, 0.999])
    torch.cuda.synchronize()
    assert 2 <= cyc[0].item() / 100_000 <= 12


@pytest.mark.parametrize("t", [16, 32])
def test_tiled_matmul_matches_torch(t):
    torch.backends.cuda.matmul.allow_tf32 = False
    n = 256
    a, b = torch.randn(n, n, device="cuda"), torch.randn(n, n, device="cuda")
    c = torch.empty_like(a)
    K.get("mm_tiled", T=t)(grid=(n // t, n // t, 1), block=(t, t, 1), args=[a, b, c, n])
    assert torch.allclose(c, a @ b, atol=1e-3, rtol=1e-4)


def test_driver_occupancy_matches_model():
    for cap in (64, 128):
        k = K.get("hungry", (f"--maxrregcount={cap}",))
        regs = K.num_regs(k)
        for tpb, smem in [(64, 0), (256, 0), (128, 57 * 1024), (512, 10_000)]:
            assert K.driver_blocks_per_sm(k, tpb, smem) == occupancy(tpb, regs, smem)["blocks"]


def test_column_access_is_slower_than_row():
    from whygpu.models import tile_access
    k = K.get("banks")
    out = torch.empty(2 * 1024, device="cuda")
    cyc = torch.zeros(2, dtype=torch.int64, device="cuda")
    res = {}
    for name, addr in (("row", tile_access("row")), ("column", tile_access("column"))):
        a = torch.tensor(addr, dtype=torch.int32, device="cuda")
        for _ in range(2):
            k(grid=(2, 1, 1), block=(1024, 1, 1), args=[a, out, cyc, 200])
        torch.cuda.synchronize()
        res[name] = cyc.float().mean().item()
    assert res["column"] > 8 * res["row"]
