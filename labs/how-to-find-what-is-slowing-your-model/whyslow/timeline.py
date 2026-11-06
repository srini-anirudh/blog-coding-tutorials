"""The timeline is the primitive.

A lane is a list of (start, end) intervals. Everything in this file is interval arithmetic on lanes:
how long a lane was busy, where its gaps are, and how much of one lane no other lane was able to hide.
Times are in whatever unit the caller uses (the trace helpers below use microseconds).
"""

from __future__ import annotations

import json
from pathlib import Path

Interval = tuple[float, float]


def merge(intervals: list[Interval]) -> list[Interval]:
    """Union of intervals, sorted and non-overlapping. Two streams running at once count once."""
    out: list[list[float]] = []
    for s, e in sorted(intervals):
        if e <= s:
            continue
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def busy_time(intervals: list[Interval]) -> float:
    return sum(e - s for s, e in merge(intervals))


def gaps(intervals: list[Interval], window: Interval) -> list[Interval]:
    """Idle stretches of a lane inside a window. The gaps are the measurement."""
    lo, hi = window
    out, t = [], lo
    for s, e in merge(intervals):
        s, e = max(s, lo), min(e, hi)
        if e <= s:
            continue
        if s > t:
            out.append((t, s))
        t = max(t, e)
    if t < hi:
        out.append((t, hi))
    return out


def busy_fraction(intervals: list[Interval], window: Interval) -> float:
    """Sum of busy time divided by wall time. 1 / this is the most any kernel work can buy."""
    lo, hi = window
    clipped = [(max(s, lo), min(e, hi)) for s, e in intervals]
    return busy_time(clipped) / (hi - lo)


def exposed(lane: list[Interval], hiders: list[Interval]) -> float:
    """Time `lane` was busy while `hiders` were not. For a collective lane and a compute lane this is
    exposed communication: the only communication number worth reporting."""
    a, b = merge(lane), merge(hiders)
    covered, j = 0.0, 0
    for s, e in a:
        while j < len(b) and b[j][1] <= s:
            j += 1
        k = j
        while k < len(b) and b[k][0] < e:
            covered += min(e, b[k][1]) - max(s, b[k][0])
            k += 1
    return busy_time(a) - covered


def max_speedup(fraction_removed: float) -> float:
    """Amdahl: removing a part that is `fraction_removed` of the step can never buy more than this."""
    return 1.0 / (1.0 - fraction_removed)


def step_time(critical_compute: float, others: dict[str, float], hidden_fraction: dict[str, float]) -> dict:
    """Lab 02 arithmetic: step = critical-path compute + the part of everything else that had nothing to hide under.

    `others` maps a resource name to its busy time and `hidden_fraction` to the share of that time that ran
    underneath compute. Hidden time cannot exceed the compute available to hide it, so the hidden parts are
    capped at `critical_compute` in total.
    """
    want = {k: others[k] * hidden_fraction.get(k, 0.0) for k in others}
    scale = min(1.0, critical_compute / sum(want.values())) if sum(want.values()) > 0 else 1.0
    exposed_parts = {k: others[k] - want[k] * scale for k in others}
    total = critical_compute + sum(exposed_parts.values())
    return {"step": total, "exposed": exposed_parts, "sum_of_parts": critical_compute + sum(others.values()),
            "max_of_parts": max([critical_compute, *others.values()])}


# ---- reading a torch.profiler chrome trace -------------------------------------------------------------------

GPU_CATS = ("kernel", "gpu_memcpy", "gpu_memset")


def load_trace(path: str | Path) -> list[dict]:
    data = json.loads(Path(path).read_text())
    return data["traceEvents"] if isinstance(data, dict) else data


def lanes_from_trace(events: list[dict]) -> dict[str, list[Interval]]:
    """Split a chrome trace into lanes: one per GPU stream, plus 'gpu' (all streams) and 'steps'
    (the ProfilerStep#N ranges the schedule writes on the CPU thread)."""
    lanes: dict[str, list[Interval]] = {"gpu": [], "steps": [], "launch": []}
    for ev in events:
        if ev.get("ph") != "X" or "dur" not in ev:
            continue
        s, e = float(ev["ts"]), float(ev["ts"]) + float(ev["dur"])
        cat = ev.get("cat", "")
        if cat in GPU_CATS:
            lanes["gpu"].append((s, e))
            lanes.setdefault(f"stream {ev.get('tid')}", []).append((s, e))
        elif cat == "cuda_runtime" and "Launch" in ev.get("name", ""):
            lanes["launch"].append((s, e))
        elif ev.get("name", "").startswith("ProfilerStep#") and cat != "gpu_user_annotation":
            # kineto also writes a GPU-side copy of each step range; count only the CPU one
            lanes["steps"].append((s, e))
    return lanes


def kernel_stats(events: list[dict]) -> dict:
    """Count, mean and median duration of GPU kernels. Thousands of kernels under ~20 us is the
    signature of the kernel-latency row of the taxonomy."""
    durs = sorted(float(ev["dur"]) for ev in events if ev.get("ph") == "X" and ev.get("cat") == "kernel")
    if not durs:
        return {"kernels": 0, "mean_us": 0.0, "median_us": 0.0}
    return {"kernels": len(durs), "mean_us": sum(durs) / len(durs), "median_us": durs[len(durs) // 2]}


def summarise(events: list[dict]) -> dict:
    """The triage numbers the post asks for first: GPU busy fraction per step, the gaps, kernel stats."""
    lanes = lanes_from_trace(events)
    steps = sorted(lanes["steps"])
    if not steps:  # no schedule: treat the GPU lane's extent as the window
        g = merge(lanes["gpu"])
        steps = [(g[0][0], g[-1][1])] if g else []
    per_step = []
    for w in steps:
        gp = gaps(lanes["gpu"], w)
        per_step.append({"wall_us": w[1] - w[0], "busy_fraction": busy_fraction(lanes["gpu"], w),
                         "largest_gap_us": max((e - s for s, e in gp), default=0.0), "n_gaps": len(gp)})
    out = {"steps": per_step, **kernel_stats(events)}
    if per_step:
        # One window over all captured steps. GPU work runs behind the CPU, so a single step's CPU range
        # can miss its own kernels when nothing synchronises; the whole window does not.
        g = merge(lanes["gpu"])
        win = (steps[0][0], max(steps[-1][1], g[-1][1] if g else steps[-1][1]))
        out["busy_fraction"] = busy_fraction(lanes["gpu"], win)
        out["step_us"] = (win[1] - win[0]) / len(steps)
        out["gpu_busy_us_per_step"] = busy_time([(max(s, win[0]), min(e, win[1])) for s, e in lanes["gpu"]]) / len(steps)
    return out
