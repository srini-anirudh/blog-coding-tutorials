"""Write results files that start with the hardware and the command that produced them."""

from __future__ import annotations

import json
import os
import platform
import sys
from pathlib import Path

import torch

LAB = Path(__file__).resolve().parents[1]
RESULTS = LAB / "results"
RUNS = LAB / "runs"  # checkpoints; ignored by git


def hardware() -> str:
    cpu = platform.processor() or platform.machine()
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip().replace(" Processor", "")
                break
    except OSError:
        pass
    return f"{cpu}, {torch.get_num_threads()} threads, torch {torch.__version__.split('+')[0]}, CPU only"


def command() -> str:
    script = Path(sys.argv[0]).resolve()
    try:
        script = script.relative_to(LAB.parents[1])
    except ValueError:
        script = Path(script.name)
    return " ".join(["python", str(script), *sys.argv[1:]])


class Report:
    """Collect printed lines, then save them as results/<name>.txt (and optional JSON) with a header."""

    def __init__(self, name: str):
        self.name, self.lines = name, []
        self.header = [f"# hardware: {hardware()}", f"# command: {command()}"]
        for h in self.header:
            print(h)

    def __call__(self, *parts):
        line = " ".join(str(p) for p in parts)
        print(line, flush=True)
        self.lines.append(line)

    def save(self, data=None):
        RESULTS.mkdir(exist_ok=True)
        (RESULTS / f"{self.name}.txt").write_text("\n".join(self.header + self.lines) + "\n")
        if data is not None:
            payload = {"hardware": hardware(), "command": command(), **data}
            (RESULTS / f"{self.name}.json").write_text(json.dumps(payload, indent=2) + "\n")


def seed_threads(n: int | None = None):
    n = n or int(os.environ.get("OMP_NUM_THREADS", "8"))
    torch.set_num_threads(n)
