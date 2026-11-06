import importlib.util
from pathlib import Path

from servekit.block_manager import BlockManager
from servekit.scheduler import Scheduler, SchedulerConfig, Sequence

spec = importlib.util.spec_from_file_location("trace05b", Path(__file__).resolve().parents[1] / "scripts" / "05b_schedule_trace.py")
trace05b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trace05b)


def test_token_budget_caps_every_iteration():
    for chunk in (128, 512):
        rows = trace05b.trace(chunk, long_len=2000, out_len=40)
        assert max(r[0] for r in rows) <= chunk
        # decodes are never starved once running: every iteration that carries the long prompt also decodes
        assert all(r[2] > 0 for r in rows if r[1] > 1 and r[0] < chunk + 100 and r is not rows[0])


def test_unchunked_prompt_is_one_iteration():
    rows = trace05b.trace(None, long_len=2000, out_len=40)
    assert max(r[0] for r in rows) == 2032  # the whole prompt plus the 32 decodes


def test_fifo_and_slot_cap():
    bm = BlockManager(1000, 16)
    sch = Scheduler(SchedulerConfig(max_num_seqs=2, max_num_batched_tokens=100), bm)
    for i in range(4):
        sch.add(Sequence(i, [1] * 10, 5))
    plan = sch.schedule()
    assert [s.seq_id for s, _ in plan] == [0, 1]  # arrival order, two slots


def test_watermark_refuses_admission_that_would_starve_growth():
    bm = BlockManager(10, 16, watermark=0.5)  # keep 5 blocks free
    sch = Scheduler(SchedulerConfig(max_num_seqs=8, max_num_batched_tokens=1000), bm)
    sch.add(Sequence(0, [1] * 64, 4))  # 4 blocks: leaves 6 free
    sch.add(Sequence(1, [1] * 64, 4))  # would leave 2 free, below the watermark
    plan = sch.schedule()
    assert [s.seq_id for s, _ in plan] == [0]
    assert len(sch.waiting) == 1 and 1 not in bm.tables
    bm.check_invariants()
