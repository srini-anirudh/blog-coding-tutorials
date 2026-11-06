"""Scale 1 of the zoom ladder: the framework profiler, wrapped so it returns the triage numbers."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, schedule

from whyslow import timeline


def profile_steps(step, active: int = 6, warmup: int = 3, wait: int = 2, trace_path: str | Path | None = None,
                  row_limit: int = 8) -> dict:
    """Run `step()` wait+warmup+active times under torch.profiler with a schedule, so warm-up is not captured.
    Returns the timeline summary of the active steps plus the top operators by self device time."""
    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if torch.cuda.is_available() else [])
    with profile(activities=acts, schedule=schedule(wait=wait, warmup=warmup, active=active),
                 record_shapes=False) as prof:
        for _ in range(wait + warmup + active):
            step()
            prof.step()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    path = Path(trace_path) if trace_path else Path(tempfile.mkstemp(suffix=".json")[1])
    path.parent.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(str(path))
    events = timeline.load_trace(path)
    if trace_path is None:
        path.unlink()
    out = timeline.summarise(events)
    sort = "self_cuda_time_total" if torch.cuda.is_available() else "self_cpu_time_total"
    out["top_ops"] = prof.key_averages().table(sort_by=sort, row_limit=row_limit, max_name_column_width=40)
    return out


def unprofiled_step_ms(step, iters: int = 20, warmup: int = 5) -> float:
    """Step time from plain wall clock, to compare with the profiled one: the profiler changes what it measures."""
    sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
    for _ in range(warmup):
        step()
    sync()
    t = time.perf_counter()
    for _ in range(iters):
        step()
    sync()
    return (time.perf_counter() - t) / iters * 1e3
