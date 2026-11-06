"""Checks that need a CUDA device. Run with: python -m pytest labs/how-to-find-what-is-slowing-your-model -m gpu"""

import importlib.util
from pathlib import Path

import pytest
import torch

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")]
LAB = Path(__file__).resolve().parent.parent


def _script(name):
    spec = importlib.util.spec_from_file_location(name, LAB / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_dense_step_is_busier_than_tiny_step():
    from whyslow.profile import profile_steps
    from whyslow.toy import ResMLP, make_step

    fr = {}
    for d, batch, dt in [(128, 32, None), (2048, 4096, torch.bfloat16)]:
        m = ResMLP(d, 4).cuda()
        x, y = torch.randn(batch, d, device="cuda"), torch.randint(0, 16, (batch,), device="cuda")
        fr[d] = profile_steps(make_step(m, x, y, dtype=dt), active=3, warmup=2, wait=1)["busy_fraction"]
    assert fr[2048] > 0.8 > fr[128]


def test_big_matmul_is_compute_bound():
    from whyslow.bench import cuda_time
    from whyslow.roofline import H100_SXM, matmul_counts, verdict

    a = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    sec = cuda_time(lambda: a @ a, iters=5)
    v = verdict(*matmul_counts(8192, 8192, 8192, 2), sec, H100_SXM["bf16_flops"], H100_SXM["hbm_bytes_per_s"])
    assert v.regime == "compute bound"


def test_fragmentation_oom_and_stable_order_fix():
    mem = _script("06_memory")
    default = mem.run_child(None)
    assert default["result"] == "OOM" and default["diagnosis"].startswith("fragmentation")
    assert default["before"]["cached_idle"] > 1 << 30  # over a GiB idle, and still no room
    assert mem.run_child(None, stable=True)["result"] == "ok"
