"""Write a script's output to results/<name>.txt (and .json), starting with the hardware and the command."""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch

LAB = Path(__file__).resolve().parents[1]
RESULTS = LAB / "results"


def hardware() -> str:
    cpu = platform.processor() or "unknown CPU"
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip().replace(" Processor", "")
                break
    except OSError:
        pass
    return (f"{cpu}, {torch.get_num_threads()} threads, CPU only; python {platform.python_version()}, "
            f"torch {torch.__version__.split('+')[0]}, numpy {np.__version__}")


def command() -> str:
    script = Path(sys.argv[0]).resolve()
    try:
        script = script.relative_to(LAB.parents[1])        # path from the repository root
    except ValueError:
        script = Path(script.name)
    return " ".join(["python", str(script)] + sys.argv[1:])


class Report:
    def __init__(self, name: str, write: bool = True):
        self.name, self.write = name, write
        self.lines = [f"# hardware: {hardware()}", f"# command: {command()}", ""]
        self.data: dict = {"hardware": hardware(), "command": command()}
        for line in self.lines[:2]:
            print(line)

    def __call__(self, *parts) -> None:
        line = " ".join(str(p) for p in parts)
        print(line)
        self.lines.append(line)

    def table(self, header: list[str], rows: list[list]) -> None:
        cells = [header] + [[f"{c:.4g}" if isinstance(c, float) else str(c) for c in r] for r in rows]
        widths = [max(len(r[i]) for r in cells) for i in range(len(header))]
        for j, r in enumerate(cells):
            self("  ".join(c.rjust(w) for c, w in zip(r, widths)))
            if j == 0:
                self("  ".join("-" * w for w in widths))

    def save(self) -> None:
        if not self.write:
            return
        RESULTS.mkdir(exist_ok=True)
        (RESULTS / f"{self.name}.txt").write_text("\n".join(self.lines) + "\n")
        (RESULTS / f"{self.name}.json").write_text(json.dumps(self.data, indent=1) + "\n")
