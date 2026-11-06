"""Run many small trainings in parallel, and keep results honest about where they came from.

Each training run is tiny, so a script that needs twenty of them runs them as separate processes with one
thread each. Workers are forked, so the data pools built in the parent are shared and not copied.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import platform
import sys
import time
from pathlib import Path
from typing import Callable, Iterable

import torch

RESULTS = Path(__file__).resolve().parents[1] / "results"


def hardware_line(workers: int | None = None) -> str:
    cpu = platform.processor() or platform.machine()
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    how = f"{workers} worker processes x 1 thread" if workers else f"{torch.get_num_threads()} threads"
    return f"# hardware: {cpu}, CPU only, {how}, torch {torch.__version__.split('+')[0]}"


def command_line() -> str:
    script = Path(sys.argv[0])
    rel = Path(*script.parts[-4:]) if len(script.parts) >= 4 else script
    return "# command: python " + " ".join([str(rel), *sys.argv[1:]])


class Report:
    """Prints to stdout and writes the same lines to results/<name>.txt, plus a JSON sidecar."""

    def __init__(self, name: str, workers: int | None = None) -> None:
        self.name = name
        self.lines = [hardware_line(workers), command_line()]
        self.data: dict = {}
        self.t0 = time.time()
        for line in self.lines:
            print(line)

    def __call__(self, *parts) -> None:
        line = " ".join(str(p) for p in parts)
        print(line, flush=True)
        self.lines.append(line)

    def table(self, header: list[str], rows: Iterable[Iterable]) -> None:
        rows = [[str(c) for c in r] for r in rows]
        widths = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h) for i, h in enumerate(header)]
        self(" | ".join(h.ljust(w) for h, w in zip(header, widths)))
        self("-+-".join("-" * w for w in widths))
        for r in rows:
            self(" | ".join(c.ljust(w) for c, w in zip(r, widths)))

    def save(self) -> None:
        self(f"# wall time: {time.time() - self.t0:.0f} s")
        RESULTS.mkdir(exist_ok=True)
        (RESULTS / f"{self.name}.txt").write_text("\n".join(self.lines) + "\n")
        if self.data:
            (RESULTS / f"{self.name}.json").write_text(json.dumps(self.data, indent=1) + "\n")


def _init_worker() -> None:
    torch.set_num_threads(1)


def n_procs() -> int:
    return int(os.environ.get("MIXLAB_PROCS", "8"))


def parallel_map(fn: Callable, items: list, procs: int | None = None) -> list:
    """map(fn, items) over forked processes, in order. procs=1 runs inline (used by the tests)."""
    procs = procs or n_procs()
    if procs <= 1 or len(items) <= 1:
        return [fn(x) for x in items]
    with mp.get_context("fork").Pool(min(procs, len(items)), initializer=_init_worker) as pool:
        return pool.map(fn, items, chunksize=1)
