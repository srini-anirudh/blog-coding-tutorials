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
    # The allocator holds the BF16 weights plus RoPE's cos/sin tables (buffers, also BF16), nothing else.
    # Its blocks are rounded up to 512 bytes, so allow a few blocks.
    assert out["weights delta bytes"] == pytest.approx(out["param bytes"] + out["buffer bytes"], abs=4096)
    assert out["param bytes"] == 2 * out["params"]
    assert out["buffer bytes"] > 0  # 2.04 bytes per parameter at this small shape, not 2.00
    # 12 bytes per parameter, plus allocator slack: when a tensor is carved from a cached block and the
    # remainder is under 1 MiB, the allocator hands out the whole block and memory_allocated counts it.
    slack = out["optimizer state delta bytes"] - 12 * out["params"]
    assert 0 <= slack <= 2 * 2**20
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
