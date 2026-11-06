"""Every parallel version must match the single-device reference. Multi-rank tests use gloo on CPU."""

import pytest
import torch

from ppbook import mesh, workers
from ppbook.dist_utils import spawn
from ppbook.pipeline import bubble_formula, gpipe, interleaved, one_f_one_b, simulate

TOL = 1e-5


@pytest.mark.parametrize("n", [2, 4])
def test_ring_allreduce_and_identity(n):
    r = spawn(workers.collectives_check, n, 256 * n)
    assert all(x["ring_vs_native"] < TOL and x["rs_then_ag_vs_allreduce"] == 0 for x in r)
    assert r[0]["ratio"] == pytest.approx(2 * (n - 1) / n)


@pytest.mark.parametrize("n", [2, 4])
def test_tensor_parallel_matches_one_device(n):
    r = spawn(workers.tp_check, n, 32, 4, 8, 2)
    for x in r:
        assert max(x["out_err"], x["dx_err"], x["grad_err"]) < TOL
    led = r[0]["ledger"]
    assert led == {"all_reduce/fwd": led["all_reduce/fwd"], "all_reduce/bwd": led["all_reduce/bwd"]}
    assert led["all_reduce/fwd"]["count"] == 2 and led["all_reduce/bwd"]["count"] == 2
    assert led["all_reduce/fwd"]["bytes"] == 2 * r[0]["act_msg_bytes"]  # each message is b*s*h


@pytest.mark.parametrize("regather", [False, True])
def test_sequence_parallel_matches_and_moves_same_bytes(regather):
    tp = spawn(workers.tp_check, 2, 32, 4, 8, 2, False)[0]
    sp = spawn(workers.tp_check, 2, 32, 4, 8, 2, True, 0, regather)[0]
    assert max(sp["out_err"], sp["dx_err"], sp["grad_err"]) < TOL
    wire = lambda led, ph: sum(v["wire"] for k, v in led.items() if k.endswith(ph))  # noqa: E731
    assert wire(sp["ledger"], "/fwd") == wire(tp["ledger"], "/fwd")
    extra = sp["ledger"]["all_gather/bwd"]["count"] - 2
    assert extra == (2 if regather else 0)  # the regather costs one AllGather per column-parallel matmul
    assert sp["activation_bytes"] < tp["activation_bytes"]


def test_row_first_needs_more_communication():
    r = spawn(workers.row_first_check, 2)[0]
    assert r["col_then_row_err"] < TOL and r["row_then_col_err"] < TOL
    assert r["gelu_of_partials_err"] > 0.01
    assert sum(v["count"] for v in r["row_then_col_fwd"].values()) == 2
    assert sum(v["count"] for v in r["col_then_row_fwd"].values()) == 1


@pytest.mark.parametrize("P,m", [(2, 2), (4, 4), (4, 8), (8, 16), (3, 7)])
def test_bubble_formula(P, m):
    for sch in ("gpipe", "1f1b"):
        assert simulate(P, m, sch)["bubble"] == pytest.approx(bubble_formula(P, m))
    if m % P == 0:
        assert simulate(P, m, "interleaved", v=2, t_f=2, t_b=4)["bubble"] == pytest.approx(bubble_formula(P, m, 2))
    assert simulate(P, m, "zb", t_f=1, t_b=1, t_w=1)["bubble"] < bubble_formula(P, m)


def test_schedule_memory_and_completeness():
    P, m = 4, 16
    assert simulate(P, m, "gpipe")["peak_stashes"] == [m] * P
    assert simulate(P, m, "1f1b")["peak_stashes"] == [P - s for s in range(P)]
    for orders in (gpipe(P, m), one_f_one_b(P, m), interleaved(P, m, 2)):
        for ops in orders:  # every forward and backward exactly once
            assert len(ops) == len(set(ops)) == 2 * m * (1 + max(c for _, _, c in ops))


@pytest.mark.parametrize("P,schedule", [(2, "gpipe"), (2, "1f1b"), (4, "1f1b")])
def test_pipeline_matches_one_device(P, schedule):
    rs = spawn(workers.pipeline_run, P, schedule, 4)
    assert max(r["grad_err"] for r in rs) < 1e-6
    assert rs[-1]["loss"] == pytest.approx(rs[-1]["ref_loss"], rel=1e-5)
    if schedule == "1f1b":
        assert [r["peak_stashes"] for r in rs] == [min(P - s, 4) for s in range(P)]


@pytest.mark.parametrize("C,layout", [(2, "contiguous"), (4, "contiguous"), (2, "zigzag"), (4, "zigzag")])
def test_context_parallel_exact(C, layout):
    rs = spawn(workers.cp_check, C, 8 * C, 4, 4, 1, layout)  # Ulysses needs heads % C == 0
    assert max(r["ring_err"] for r in rs) < TOL
    blocks = [r["blocks"] for r in rs]
    assert blocks == ([2 * C + 1] * C if layout == "zigzag" else list(range(1, C + 1)))
    if layout == "contiguous":
        assert max(max(r["ulysses_err"], r["ulysses_grad_err"]) for r in rs) < TOL
        assert rs[0]["ulysses_ledger"]["all_to_all/fwd"]["count"] == 4


def test_mesh():
    for r in range(512):
        assert mesh.rank_of(mesh.coords(r, 8, 1, 8, 8), 8, 1, 8, 8) == r
    rep = mesh.placement_report(8, 1, 8, 8)
    assert rep["tp"]["cross_node"] == 0 and rep["dp"]["cross_node"] == rep["dp"]["groups"]
    assert mesh.placement_report(16, 1, 4, 8)["tp"]["cross_node"] == 32


def test_fake_comm_keeps_shapes():
    from ppbook import comm
    with comm.fake_comm():
        assert comm.all_reduce(torch.ones(3)).shape == (3,)
