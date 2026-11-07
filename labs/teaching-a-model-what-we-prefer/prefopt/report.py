"""Write a script's output to results/<name>.txt and results/<name>.json.

Every file starts with the hardware it ran on and the command that wrote it.
"""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import torch

LAB = Path(__file__).resolve().parents[1]
RESULTS = LAB / "results"


def hardware() -> str:
    cpu = platform.processor() or platform.machine()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip().replace(" Processor", "")
                break
    except OSError:
        pass
    return f"{cpu}, {torch.get_num_threads()} threads, torch {torch.__version__.split('+')[0]}, CPU only"


def command() -> str:
    script = Path(sys.argv[0]).resolve()
    try:
        script = script.relative_to(LAB.parents[1])  # the repository root
    except ValueError:
        script = Path(script.name)
    return " ".join(["python", str(script), *sys.argv[1:]])


class Report:
    """Collect printed lines and a JSON dict; save both with a header."""

    def __init__(self, name: str):
        self.name, self.lines, self.data = name, [], {}
        self.header = f"# hardware: {hardware()}\n# command: {command()}"
        print(self.header)

    def line(self, text: str = "") -> None:
        print(text, flush=True)
        self.lines.append(text)

    def table(self, header: list[str], rows: list[list]) -> None:
        cells = [header] + [[_fmt(c) for c in r] for r in rows]
        widths = [max(len(r[i]) for r in cells) for i in range(len(header))]
        for i, r in enumerate(cells):
            self.line("  ".join(c.rjust(w) for c, w in zip(r, widths)))
            if i == 0:
                self.line("  ".join("-" * w for w in widths))

    def save(self) -> None:
        RESULTS.mkdir(exist_ok=True)
        (RESULTS / f"{self.name}.txt").write_text(self.header + "\n" + "\n".join(self.lines) + "\n")
        payload = {"hardware": hardware(), "command": command(), **self.data}
        (RESULTS / f"{self.name}.json").write_text(json.dumps(payload, indent=1) + "\n")


def _fmt(c) -> str:
    if isinstance(c, float):
        c = c + 0.0  # -0.0 prints as 0
        return f"{c:.4f}" if (1e-3 <= abs(c) < 1e4 or c == 0) else f"{c:.2e}"
    return str(c)
