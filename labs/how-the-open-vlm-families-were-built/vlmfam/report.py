"""Hardware labels and result files. Every file in results/ starts with a hardware line and the command."""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
RESULTS = LAB / "results"
DATA = LAB / "data"


def cpu_name() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].replace(" Processor", "").strip()
    return platform.processor() or "CPU"


def hardware() -> str:
    import torch
    import transformers

    return (f"{cpu_name()}, {torch.get_num_threads()} threads, torch {torch.__version__.split('+')[0]}, "
            f"transformers {transformers.__version__}")


def command() -> str:
    """The command that ran, relative to the repository root."""
    script = Path(sys.argv[0]).resolve()
    try:
        script = script.relative_to(LAB.parents[1])
    except ValueError:
        script = Path(script.name)
    return " ".join(["python", str(script), *sys.argv[1:]])


class Report:
    """Collects printed lines and writes them, with a header, to results/<name>.txt (and optional JSON)."""

    def __init__(self, name: str):
        self.name = name
        self.lines = [f"# hardware: {hardware()}", f"# command: {command()}", ""]
        print("\n".join(self.lines[:2]))

    def __call__(self, *parts) -> None:
        line = " ".join(str(p) for p in parts)
        print(line)
        self.lines.append(line)

    def table(self, header: list[str], rows: list[list]) -> None:
        cells = [[str(c) for c in header]] + [[str(c) for c in r] for r in rows]
        widths = [max(len(r[i]) for r in cells) for i in range(len(header))]
        for j, r in enumerate(cells):
            self("  ".join(c.ljust(w) if i == 0 else c.rjust(w) for i, (c, w) in enumerate(zip(r, widths))))
            if j == 0:
                self("  ".join("-" * w for w in widths))

    def save(self, data: dict | list | None = None) -> None:
        RESULTS.mkdir(exist_ok=True)
        (RESULTS / f"{self.name}.txt").write_text("\n".join(self.lines) + "\n")
        if data is not None:
            payload = {"hardware": hardware(), "command": command(), "data": data}
            (RESULTS / f"{self.name}.json").write_text(json.dumps(payload, indent=1) + "\n")
