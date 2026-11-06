"""Shared helpers for the scripts: import path, hardware line, and writing results."""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import torch

LAB = Path(__file__).resolve().parents[1]
RESULTS = LAB / "results"
sys.path.insert(0, str(LAB))


def hardware() -> str:
    """CPU model and thread count, plus torch version. Never a hostname."""
    model = platform.processor() or "unknown CPU"
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return f"{model}, {torch.get_num_threads()} threads, torch {torch.__version__}"


class Report:
    """Collects printed lines and a JSON payload, then writes results/<name>.txt and .json."""

    def __init__(self, name: str):
        self.name = name
        script = Path(sys.argv[0]).name
        args = " ".join(sys.argv[1:])
        self.lines = [f"# hardware: {hardware()}",
                      f"# command: python labs/seeing-in-time/scripts/{script} {args}".rstrip()]
        self.data: dict = {"hardware": hardware()}

    def print(self, *parts) -> None:
        line = " ".join(str(p) for p in parts)
        print(line, flush=True)
        self.lines.append(line)

    def table(self, header: list[str], rows: list[list]) -> None:
        widths = [max(len(str(x)) for x in col) for col in zip(header, *rows)]
        fmt = "  ".join(f"{{:<{w}}}" for w in widths)
        self.print(fmt.format(*header))
        for r in rows:
            self.print(fmt.format(*[str(x) for x in r]))

    def save(self) -> None:
        RESULTS.mkdir(exist_ok=True)
        (RESULTS / f"{self.name}.txt").write_text("\n".join(self.lines) + "\n")
        (RESULTS / f"{self.name}.json").write_text(json.dumps(self.data, indent=1, default=str) + "\n")
