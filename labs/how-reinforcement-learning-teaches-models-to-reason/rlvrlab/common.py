"""Hardware labels and result files. Every file in results/ starts with the hardware line and the command."""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch

LAB = Path(__file__).resolve().parent.parent
REPO = LAB.parent.parent
RESULTS = LAB / "results"


def cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].replace("Processor", "").strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def hardware_line() -> str:
    return (f"Hardware: {cpu_name()}, {torch.get_num_threads()} threads, CPU only. "
            f"torch {torch.__version__.split('+')[0]}, numpy {np.__version__}")


def command_line() -> str:
    script = Path(sys.argv[0]).resolve()
    try:
        script = script.relative_to(REPO)
    except ValueError:
        script = Path(script.name)
    return "Command: python " + " ".join([str(script), *sys.argv[1:]])


class Report:
    """Collect printed lines and a JSON payload, then write results/<name>.txt and results/<name>.json."""

    def __init__(self, name: str):
        self.name = name
        self.lines = [hardware_line(), command_line(), ""]
        self.data: dict = {}
        for line in self.lines[:2]:
            print(line)
        print()

    def print(self, *parts) -> None:
        line = " ".join(str(p) for p in parts)
        print(line)
        self.lines.append(line)

    def table(self, header: list[str], rows: list[list]) -> None:
        cells = [[str(h) for h in header]] + [[f"{c:.4g}" if isinstance(c, float) else str(c) for c in r] for r in rows]
        widths = [max(len(r[i]) for r in cells) for i in range(len(header))]
        for j, r in enumerate(cells):
            self.print("  ".join(c.rjust(w) for c, w in zip(r, widths)))
            if j == 0:
                self.print("  ".join("-" * w for w in widths))

    def save(self) -> None:
        RESULTS.mkdir(exist_ok=True)
        (RESULTS / f"{self.name}.txt").write_text("\n".join(self.lines) + "\n")
        payload = {"hardware": self.lines[0], "command": self.lines[1], **self.data}
        (RESULTS / f"{self.name}.json").write_text(json.dumps(payload, indent=1) + "\n")
