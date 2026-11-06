"""Latency metrics, goodput, synthetic workloads and the header every results file starts with."""

from __future__ import annotations

import json
import os
import platform
import random
import sys
from pathlib import Path

import torch

LAB = Path(__file__).resolve().parents[1]
RESULTS = LAB / "results"


def hardware_line() -> str:
    if torch.cuda.is_available():
        return f"{torch.cuda.get_device_name(0)}, torch {torch.__version__}, CUDA {torch.version.cuda}"
    name = platform.processor() or "unknown CPU"
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                name = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return f"{name}, {torch.get_num_threads()} threads, torch {torch.__version__}"


def command_line() -> str:
    script = Path(sys.argv[0]).resolve()
    try:
        rel = script.relative_to(LAB.parents[1])  # path from the repository root
    except ValueError:
        rel = Path(script.name)
    return " ".join(["python", str(rel), *sys.argv[1:]])


class Report:
    """Print and also write to results/<name>; the first two lines say where and how it was made."""

    def __init__(self, name: str) -> None:
        RESULTS.mkdir(exist_ok=True)
        self.path = RESULTS / name
        self.lines = [f"# hardware: {hardware_line()}", f"# command: {command_line()}"]
        for line in self.lines:
            print(line)

    def __call__(self, *parts) -> None:
        line = " ".join(str(p) for p in parts)
        print(line, flush=True)
        self.lines.append(line)

    def table(self, rows: list[dict], fmt: dict | None = None) -> None:
        if not rows:
            return
        cols = list(rows[0])
        fmt = fmt or {}
        cells = [[fmt.get(c, "{}").format(r[c]) if not isinstance(r[c], str) else r[c] for c in cols] for r in rows]
        widths = [max(len(c), *(len(row[i]) for row in cells)) for i, c in enumerate(cols)]
        self("  ".join(c.rjust(w) for c, w in zip(cols, widths)))
        for row in cells:
            self("  ".join(v.rjust(w) for v, w in zip(row, widths)))

    def save(self, data: dict | None = None) -> None:
        self.path.write_text("\n".join(self.lines) + "\n")
        if data is not None:
            meta = {"hardware": hardware_line(), "command": command_line()}
            self.path.with_suffix(".json").write_text(json.dumps({**meta, **data}, indent=1) + "\n")


def percentile(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    k = min(len(s) - 1, max(0, round(q / 100 * (len(s) - 1))))
    return s[k]


def request_metrics(seq) -> dict:
    """TTFT: arrival to first token. TPOT: mean gap between later tokens. max_itl: the worst single gap."""
    t = seq.token_times
    gaps = [b - a for a, b in zip(t, t[1:])]
    return {
        "ttft": t[0] - seq.arrival,
        "tpot": (t[-1] - t[0]) / (len(t) - 1) if len(t) > 1 else 0.0,
        "max_itl": max(gaps) if gaps else 0.0,
        "latency": t[-1] - seq.arrival,
        "n": len(t),
    }


def summarize(seqs, wall: float, ttft_slo: float | None = None, tpot_slo: float | None = None) -> dict:
    ms = [request_metrics(s) for s in seqs]
    out = {
        "requests": len(ms),
        "wall_s": wall,
        "req_per_s": len(ms) / wall,
        "out_tok_per_s": sum(m["n"] for m in ms) / wall,
        "ttft_p50": percentile([m["ttft"] for m in ms], 50),
        "ttft_p99": percentile([m["ttft"] for m in ms], 99),
        "tpot_p50": percentile([m["tpot"] for m in ms], 50),
        "tpot_p99": percentile([m["tpot"] for m in ms], 99),
        "latency_mean": sum(m["latency"] for m in ms) / max(1, len(ms)),
    }
    if ttft_slo is not None and tpot_slo is not None:
        ok = [m for m in ms if m["ttft"] <= ttft_slo and m["tpot"] <= tpot_slo]
        out["slo_attainment"] = len(ok) / max(1, len(ms))
        out["goodput_req_per_s"] = len(ok) / wall
    return out


def random_prompts(n: int, lo: int, hi: int, vocab: int, seed: int = 0) -> list[list[int]]:
    rng = random.Random(seed)
    return [[rng.randrange(1, vocab) for _ in range(rng.randint(lo, hi))] for _ in range(n)]


def output_lengths(n: int, lo: int, hi: int, seed: int = 0) -> list[int]:
    """Heavy-ish spread: log-uniform between lo and hi, so a yes/no answer and a long one share the queue."""
    rng = random.Random(seed + 1)
    return [int(round(lo * (hi / lo) ** rng.random())) for _ in range(n)]


def poisson_arrivals(n: int, rate: float, seed: int = 0) -> list[float]:
    rng = random.Random(seed + 2)
    t, out = 0.0, []
    for _ in range(n):
        t += rng.expovariate(rate)
        out.append(t)
    return out


def device_and_dtype(name: str | None = None):
    dev = name or ("cuda" if torch.cuda.is_available() else "cpu")
    return dev, (torch.bfloat16 if dev == "cuda" else torch.float32)


def threads_note() -> str:
    return f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', 'unset')}"
