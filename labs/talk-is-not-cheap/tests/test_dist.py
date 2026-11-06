"""Run the schedules for real on 4 gloo processes. Takes a few seconds because processes are spawned."""

import pytest
import torch

from commcost import dist_run, workers


def test_gloo_schedules_give_exact_sums():
    rows = dist_run.launch(4, workers.allreduce_bench, ([256, 4096], ["naive", "tree", "ring", "rabenseifner",
                                                                       "library"], 1, "gloo"))
    assert len(rows) == 10
    assert all(r["correct"] for r in rows)


def test_gloo_pingpong_returns_positive_times():
    rows = dist_run.launch(2, workers.pingpong, ([8, 1 << 16], 3, "gloo"))
    assert [r["bytes"] for r in rows] == [8, 1 << 16]
    assert all(r["one_way_s"] > 0 for r in rows)


@pytest.mark.gpu
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 GPUs")
def test_nccl_schedules_give_exact_sums():
    rows = dist_run.launch(2, workers.allreduce_bench, ([1 << 20], ["ring", "tree", "library"], 1, "nccl"),
                           backend="nccl")
    assert all(r["correct"] for r in rows)
