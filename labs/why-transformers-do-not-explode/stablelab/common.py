"""Small helpers shared by every script: seeding, a hardware line, and a result file writer."""

from __future__ import annotations

import platform
import random
import sys
from pathlib import Path

import numpy as np
import torch

LAB = Path(__file__).resolve().parents[1]
RESULTS = LAB / "results"
SLUG = LAB.name


def seed_all(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return torch.Generator().manual_seed(seed)


def cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip().replace(" Processor", "")
    except OSError:
        pass
    return platform.processor() or platform.machine()


def hardware_line() -> str:
    """CPU model and thread count, plus the GPU if one is visible. Never a hostname."""
    text = f"Hardware: {cpu_name()}, {torch.get_num_threads()} threads, torch {torch.__version__}"
    if torch.cuda.is_available():
        text += f", {torch.cuda.get_device_name(0)}, CUDA {torch.version.cuda}"
    return text


def command_line() -> str:
    """The command that produced a result, written relative to the repository root."""
    script = Path(sys.argv[0]).name
    return " ".join(["python", f"labs/{SLUG}/scripts/{script}", *sys.argv[1:]])


class Report:
    """Print lines as they are produced and save them to results/<name>.txt at the end."""

    def __init__(self, name: str) -> None:
        self.path = RESULTS / f"{name}.txt"
        self.lines = [hardware_line(), f"Command: {command_line()}", ""]
        for line in self.lines[:2]:
            print(line)

    def __call__(self, line: str = "") -> None:
        print(line, flush=True)
        self.lines.append(line)

    def table(self, header: list[str], rows: list[list]) -> None:
        cells = [header] + [[fmt(c) for c in row] for row in rows]
        widths = [max(len(r[i]) for r in cells) for i in range(len(header))]
        for j, row in enumerate(cells):
            self("  ".join(c.rjust(w) for c, w in zip(row, widths)))
            if j == 0:
                self("  ".join("-" * w for w in widths))

    def save(self) -> Path:
        RESULTS.mkdir(exist_ok=True)
        self.path.write_text("\n".join(self.lines) + "\n")
        print(f"\nwrote {self.path.relative_to(LAB.parents[1])}")
        return self.path


def fmt(x) -> str:
    if isinstance(x, float):
        if x != x or x in (float("inf"), float("-inf")):
            return str(x)
        if x == 0 or 1e-3 <= abs(x) < 1e5:
            return f"{x:.4g}"
        return f"{x:.3e}"
    return str(x)
