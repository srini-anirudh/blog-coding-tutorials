"""The same checks on NCCL. Needs two CUDA GPUs; skipped otherwise."""

import pytest
import torch

from ppbook import workers
from ppbook.dist_utils import spawn

pytestmark = [pytest.mark.gpu,
              pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two CUDA GPUs")]


def test_tp_sp_on_nccl():
    for sp in (False, True):
        r = spawn(workers.tp_check, 2, 64, 8, 16, 2, sp, backend="nccl")[0]
        assert max(r["out_err"], r["dx_err"], r["grad_err"]) < 1e-4


def test_pipeline_on_nccl():
    rs = spawn(workers.pipeline_run, 2, "1f1b", 4, backend="nccl")
    assert max(r["grad_err"] for r in rs) < 1e-5


def test_ring_and_ulysses_on_nccl():
    rs = spawn(workers.cp_check, 2, 32, 4, 8, 1, "contiguous", backend="nccl")
    assert max(max(r["ring_err"], r["ulysses_err"]) for r in rs) < 1e-4
