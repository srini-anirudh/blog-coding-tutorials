"""Write script output to the terminal and to `results/<name>.txt`, headed by the hardware and the command."""

from __future__ import annotations

import platform
import sys
from pathlib import Path

import torch

RESULTS = Path(__file__).resolve().parent.parent / "results"


def hardware() -> str:
    cpu = platform.processor() or platform.machine()
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip().removesuffix(" Processor")
                break
    except OSError:
        pass
    return f"{cpu}, {torch.get_num_threads()} threads, torch {torch.__version__.split('+')[0]}"


class Report:
    """Collect lines, print them as they come, and save them when the script ends."""

    def __init__(self, name: str):
        self.path = RESULTS / f"{name}.txt"
        script = Path(sys.argv[0]).resolve()
        lab = Path(__file__).resolve().parent.parent
        rel = script.relative_to(lab) if script.is_relative_to(lab) else script.name
        args = " ".join(sys.argv[1:])
        self.lines = [f"# Hardware: {hardware()}", f"# Command: python labs/{lab.name}/{rel} {args}".rstrip(), ""]
        print("\n".join(self.lines))

    def __call__(self, *parts) -> None:
        line = " ".join(str(p) for p in parts)
        print(line)
        self.lines.append(line)

    def table(self, header: list[str], rows: list[list]) -> None:
        cells = [header] + [[str(c) for c in r] for r in rows]
        widths = [max(len(r[i]) for r in cells) for i in range(len(header))]
        for i, r in enumerate(cells):
            self("  ".join(c.ljust(w) for c, w in zip(r, widths)).rstrip())
            if i == 0:
                self("  ".join("-" * w for w in widths))

    def save(self) -> None:
        RESULTS.mkdir(exist_ok=True)
        self.path.write_text("\n".join(self.lines) + "\n")
