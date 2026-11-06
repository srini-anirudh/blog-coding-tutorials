"""Write results files that start with the hardware and the command that produced them."""

from __future__ import annotations

import json
import os
import platform
import sys
from pathlib import Path

import torch

LAB = Path(__file__).resolve().parent.parent
RESULTS = Path(os.environ.get("RLFP_RESULTS", LAB / "results"))  # override to write a trial run elsewhere


def hardware() -> str:
    if torch.cuda.is_available():
        return (f"{torch.cuda.get_device_name(0)}, torch {torch.__version__}, CUDA {torch.version.cuda}")
    name = platform.processor() or "CPU"
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                name = line.split(":", 1)[1].strip().replace(" Processor", "")
                break
    except OSError:
        pass
    return f"{name}, {torch.get_num_threads()} threads, torch {torch.__version__}"


def command() -> str:
    script = Path(sys.argv[0])
    try:
        script = script.resolve().relative_to(LAB.parent.parent)
    except ValueError:
        script = Path(script.name)
    return " ".join(["python", str(script), *sys.argv[1:]])


class Report:
    """Collects printed lines and a JSON payload; `save()` writes results/<name>.txt and results/<name>.json."""

    def __init__(self, name: str):
        self.name = name
        self.lines = [f"hardware: {hardware()}", f"command: {command()}"]
        self.data: dict = {"hardware": hardware(), "command": command()}
        for line in self.lines:
            print(line)

    def print(self, *parts) -> None:
        line = " ".join(str(p) for p in parts)
        print(line, flush=True)
        self.lines.append(line)

    def table(self, rows: list[dict], cols: list[str] | None = None, fmt: str = "{:.4g}") -> None:
        cols = cols or list(rows[0])
        cells = [[c for c in cols]] + [[fmt.format(r[c]) if isinstance(r[c], float) else str(r[c]) for c in cols]
                                       for r in rows]
        widths = [max(len(row[i]) for row in cells) for i in range(len(cols))]
        for row in cells:
            self.print("  ".join(v.rjust(w) for v, w in zip(row, widths)))

    def save(self) -> None:
        RESULTS.mkdir(exist_ok=True)
        (RESULTS / f"{self.name}.txt").write_text("\n".join(self.lines) + "\n")
        (RESULTS / f"{self.name}.json").write_text(json.dumps(self.data, indent=1, default=float) + "\n")
