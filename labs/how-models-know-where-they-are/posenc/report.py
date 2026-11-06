"""Write each script's output to the terminal and to results/<name>.txt, headed by hardware and command."""

from __future__ import annotations

import json
import os
import platform
import sys
from pathlib import Path

import torch

LAB = Path(__file__).resolve().parents[1]
RESULTS = LAB / "results"


def hardware() -> str:
    cpu = platform.processor() or "unknown CPU"
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    cpu = cpu.replace(" 64-Core Processor", "")
    return f"{cpu}, {torch.get_num_threads()} threads, torch {torch.__version__.split('+')[0]}, CPU only"


def command() -> str:
    script = Path(sys.argv[0]).resolve()
    try:
        rel = script.relative_to(LAB.parents[1])
    except ValueError:
        rel = Path(script.name)
    return " ".join(["python", str(rel), *sys.argv[1:]])


class Report:
    def __init__(self, name: str) -> None:
        RESULTS.mkdir(exist_ok=True)
        self.name = name
        self.lines: list[str] = []
        self.data: dict = {"hardware": hardware(), "command": command()}
        self.print(f"hardware: {self.data['hardware']}")
        self.print(f"command: {self.data['command']}")

    def print(self, *parts) -> None:
        line = " ".join(str(p) for p in parts)
        print(line, flush=True)
        self.lines.append(line)

    def table(self, header: list[str], rows: list[list]) -> None:
        cells = [[str(h) for h in header]] + [[f"{c:.4g}" if isinstance(c, float) else str(c) for c in r] for r in rows]
        widths = [max(len(r[i]) for r in cells) for i in range(len(header))]
        for j, r in enumerate(cells):
            self.print("  ".join(c.rjust(w) for c, w in zip(r, widths)))
            if j == 0:
                self.print("  ".join("-" * w for w in widths))

    def save(self) -> None:
        (RESULTS / f"{self.name}.txt").write_text("\n".join(self.lines) + "\n")
        if len(self.data) > 2:
            (RESULTS / f"{self.name}.json").write_text(json.dumps(self.data, indent=1) + "\n")


def threads_from_env(default: int = 8) -> None:
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", default)))
