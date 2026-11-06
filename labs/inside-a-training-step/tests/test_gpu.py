"""CUDA paths at a small shape. Skipped without a GPU."""

from dataclasses import replace

import pytest
import torch

from stepanatomy import accounting as A
from stepanatomy import gpustep as G

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")]

SMALL = replace(A.LLAMA2_7B, layers=2, hidden=512, heads=4, intermediate=1376, vocab=1024, seq=512)


def test_allocator_ledger_and_timing():
    out = G.step_report(SMALL, "none", steps=2, warmup=1)
    assert out["bytes/param weights (bf16)"] == pytest.approx(2, abs=0.1)
    assert out["bytes/param master+m+v"] == pytest.approx(12, abs=0.1)
    assert out["step ms"] > 0
    curve = dict((k, v) for k, v in out["curve"])
    assert curve["the turn: loss computed"] > curve["start of step"]


def test_checkpointing_lowers_peak():
    none = G.step_report(SMALL, "none", steps=1, warmup=1)
    full = G.step_report(SMALL, "full", steps=1, warmup=1)
    assert full["activation+workspace peak GiB"] < none["activation+workspace peak GiB"]


def test_op_table_runs():
    rows = G.op_table(SMALL, compile_elementwise=False)
    assert len(rows) == 11 and all(r["ms"] > 0 for r in rows)
