"""Write a script's output to stdout and to results/<name>.txt, headed by the hardware and the command."""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parents[1] / "results"


def hardware() -> str:
    cpu = platform.processor() or "unknown CPU"
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip().replace(" Processor", "")
                break
    except OSError:
        pass
    threads = os.environ.get("OMP_NUM_THREADS", str(os.cpu_count()))
    return f"{cpu}, {threads} threads, Python {platform.python_version()}, numpy {np.__version__}"


class Report:
    def __init__(self, name: str):
        RESULTS.mkdir(exist_ok=True)
        self.path = RESULTS / f"{name}.txt"
        self.lines: list[str] = []
        script = Path(sys.argv[0]).resolve()
        rel = "/".join(script.parts[script.parts.index("labs"):]) if "labs" in script.parts else script.name
        self.add(f"hardware: {hardware()}")
        self.add(f"command: python {rel} {' '.join(sys.argv[1:])}".rstrip())
        self.add("")

    def add(self, text: str = "") -> None:
        print(text)
        self.lines.append(text)

    def table(self, header: list[str], rows: list[list], fmt: str = "{:.3f}") -> None:
        cells = [[c if isinstance(c, str) else fmt.format(c) for c in r] for r in rows]
        widths = [max(len(h), *(len(r[i]) for r in cells)) for i, h in enumerate(header)]
        self.add("  ".join(h.ljust(w) for h, w in zip(header, widths)))
        for r in cells:
            self.add("  ".join(c.ljust(w) for c, w in zip(r, widths)))
        self.add()

    def save(self) -> None:
        self.path.write_text("\n".join(self.lines) + "\n")
