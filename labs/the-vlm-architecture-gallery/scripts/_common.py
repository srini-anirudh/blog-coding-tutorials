"""Shared setup for the scripts: import path, hardware label, and result files with a header."""

from __future__ import annotations

import json
import os
import platform
import sys
from pathlib import Path

import torch

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))
RESULTS = LAB / "results"
RUNS = LAB / "runs"          # cached pretrained tiny LMs; not in git


def hardware() -> str:
    name = platform.processor() or "cpu"
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    name = line.split(":", 1)[1].strip().replace(" Processor", "")
                    break
    except OSError:
        pass
    return f"{name}, {torch.get_num_threads()} threads, torch {torch.__version__.split('+')[0]}, float32"


def command() -> str:
    script = Path(sys.argv[0]).resolve().relative_to(LAB.parents[1])
    return " ".join(["python", str(script), *sys.argv[1:]])


class Report:
    """Print and keep lines; save as results/<name>.txt with a hardware + command header."""

    def __init__(self, name: str):
        self.name = name
        self.lines = [f"# hardware: {hardware()}", f"# command: {command()}"]
        for line in self.lines:
            print(line)

    def __call__(self, line: str = ""):
        print(line, flush=True)
        self.lines.append(line)

    def save(self, data: dict | list | None = None):
        RESULTS.mkdir(exist_ok=True)
        (RESULTS / f"{self.name}.txt").write_text("\n".join(self.lines) + "\n")
        if data is not None:
            payload = {"hardware": hardware(), "command": command(), "data": data}
            (RESULTS / f"{self.name}.json").write_text(json.dumps(payload, indent=1) + "\n")


def threads(n: int | None = None):
    n = n or int(os.environ.get("OMP_NUM_THREADS", "4"))
    torch.set_num_threads(n)
