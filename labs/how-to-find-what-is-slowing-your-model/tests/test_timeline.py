import pytest
import torch

from whyslow import timeline as tl


def test_merge_counts_overlap_once():
    assert tl.merge([(5, 7), (0, 2), (1, 3), (7, 8)]) == [(0, 3), (5, 8)]
    assert tl.busy_time([(0, 2), (1, 3), (5, 8)]) == 6


def test_gaps_and_busy_fraction():
    lane = [(10, 20), (30, 40)]
    assert tl.gaps(lane, (0, 50)) == [(0, 10), (20, 30), (40, 50)]
    assert tl.busy_fraction(lane, (0, 50)) == pytest.approx(0.4)
    assert tl.busy_fraction(lane, (15, 35)) == pytest.approx(0.5)  # clipped to the window


def test_figure_01_numbers():
    # 120 ms step, GPU busy 64 ms: kernel work can buy at most 120/64.
    gpu = [(0, 30), (52, 86)]
    assert tl.busy_fraction(gpu, (0, 120)) == pytest.approx(64 / 120)
    # an all-reduce lane busy 44 ms, 28 ms of it under backward compute: 16 ms exposed
    compute = [(0, 28)]
    allreduce = [(0, 44)]
    assert tl.exposed(allreduce, compute) == pytest.approx(16)


def test_exposed_with_many_intervals():
    comm = [(0, 10), (15, 25), (40, 50)]
    comp = [(5, 20), (45, 60)]
    # hidden: 5..10 (5), 15..20 (5), 45..50 (5)
    assert tl.exposed(comm, comp) == pytest.approx(30 - 15)
    assert tl.exposed(comm, []) == pytest.approx(30)


def test_amdahl_numbers_from_the_post():
    assert tl.max_speedup(0.30) == pytest.approx(1.43, abs=0.005)
    assert 1 / 0.55 == pytest.approx(1.8, abs=0.02)
    assert 380 / 166 == pytest.approx(2.29, abs=0.005)


def test_step_time_between_max_and_sum():
    no = tl.step_time(100, {"comm": 40, "data": 20}, {})
    full = tl.step_time(100, {"comm": 40, "data": 20}, {"comm": 1.0, "data": 1.0})
    assert no["step"] == no["sum_of_parts"] == 160
    assert full["step"] == full["max_of_parts"] == 100
    # hiding is capped by the compute available to hide under
    capped = tl.step_time(30, {"comm": 40, "data": 20}, {"comm": 1.0, "data": 1.0})
    assert capped["step"] == pytest.approx(30 + 60 - 30)


def _ev(cat, ts, dur, name="k", tid=7):
    return {"ph": "X", "cat": cat, "ts": ts, "dur": dur, "name": name, "tid": tid}


def test_summarise_synthetic_trace():
    events = [_ev("user_annotation", 0, 100, "ProfilerStep#1"), _ev("user_annotation", 100, 100, "ProfilerStep#2"),
              _ev("kernel", 10, 30), _ev("kernel", 50, 10), _ev("kernel", 120, 60, tid=8),
              _ev("cuda_runtime", 5, 4, "cudaLaunchKernel")]
    s = tl.summarise(events)
    assert s["kernels"] == 3
    assert [p["busy_fraction"] for p in s["steps"]] == pytest.approx([0.4, 0.6])
    assert s["busy_fraction"] == pytest.approx(0.5)
    assert s["steps"][0]["largest_gap_us"] == pytest.approx(40)
    lanes = tl.lanes_from_trace(events)
    assert "stream 7" in lanes and "stream 8" in lanes and len(lanes["launch"]) == 1


def test_profile_steps_on_cpu_returns_top_ops():
    from whyslow.profile import profile_steps, unprofiled_step_ms

    x = torch.randn(64, 64)
    s = profile_steps(lambda: x @ x, active=2, warmup=1, wait=1)
    assert len(s["steps"]) == 2 and "aten::mm" in s["top_ops"]
    assert unprofiled_step_ms(lambda: x @ x, iters=3, warmup=1) > 0


def test_gpu_side_step_annotations_are_not_counted_twice():
    events = [_ev("user_annotation", 0, 100, "ProfilerStep#1"), _ev("gpu_user_annotation", 5, 100, "ProfilerStep#1"),
              _ev("kernel", 10, 30)]
    assert len(tl.lanes_from_trace(events)["steps"]) == 1
