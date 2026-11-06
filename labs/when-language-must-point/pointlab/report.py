"""Print a result and save the same text to results/, headed by the hardware and the command."""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import torch

LAB = Path(__file__).resolve().parents[1]
RESULTS = LAB / "results"


def hardware() -> str:
    cpu = platform.processor() or "CPU"
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip().replace(" Processor", "")
                break
    except OSError:
        pass
    dev = f"{cpu}, {torch.get_num_threads()} threads"
    if torch.cuda.is_available():
        dev = f"{torch.cuda.get_device_name(0)}, CUDA {torch.version.cuda}"
    return f"{dev}, torch {torch.__version__.split('+')[0]}"


def command() -> str:
    script = Path(sys.argv[0]).resolve()
    try:
        rel = script.relative_to(LAB.parents[1])
    except ValueError:
        rel = Path(script.name)
    return " ".join(["python", str(rel), *sys.argv[1:]])


class Report:
    def __init__(self, name: str):
        self.name = name
        self.lines = [f"# hardware: {hardware()}", f"# command: {command()}", ""]

    def __call__(self, *parts) -> None:
        text = " ".join(str(p) for p in parts)
        print(text, flush=True)
        self.lines.append(text)

    def table(self, header: list[str], rows: list[list]) -> None:
        cells = [header] + [[str(c) for c in r] for r in rows]
        widths = [max(len(r[i]) for r in cells) for i in range(len(header))]
        for k, r in enumerate(cells):
            self("  ".join(c.ljust(w) for c, w in zip(r, widths)).rstrip())
            if k == 0:
                self("  ".join("-" * w for w in widths))

    def save(self, data: dict | None = None) -> None:
        RESULTS.mkdir(exist_ok=True)
        (RESULTS / f"{self.name}.txt").write_text("\n".join(self.lines) + "\n")
        if data is not None:
            payload = {"hardware": hardware(), "command": command(), **data}
            (RESULTS / f"{self.name}.json").write_text(json.dumps(payload, indent=1) + "\n")
