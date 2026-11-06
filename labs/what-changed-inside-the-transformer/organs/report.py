"""Print a script's output and save it to results/<name>.txt, headed by the hardware line and the command."""

from __future__ import annotations

import platform
import sys
from pathlib import Path

import torch

from ._paths import LAB, RESULTS


def hardware() -> str:
    if torch.cuda.is_available():
        return f"{torch.cuda.get_device_name(0)}, torch {torch.__version__.split('+')[0]}, CUDA {torch.version.cuda}"
    cpu = "CPU"
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip().removesuffix(" Processor")
                break
    except OSError:
        cpu = platform.processor() or "CPU"
    return f"{cpu}, {torch.get_num_threads()} threads, torch {torch.__version__.split('+')[0]}"


class Report:
    def __init__(self, name: str) -> None:
        self.name = name
        script = Path(sys.argv[0]).resolve()
        try:
            rel = script.relative_to(LAB.parents[1])
        except ValueError:
            rel = Path(script.name)
        cmd = " ".join(["python", str(rel), *sys.argv[1:]])
        self.lines = [f"hardware: {hardware()}", f"command: {cmd}", ""]

    def __call__(self, *parts: object) -> None:
        line = " ".join(str(p) for p in parts)
        print(line, flush=True)
        self.lines.append(line)

    def table(self, header: list[str], rows: list[list[object]]) -> None:
        cells = [header] + [[str(c) for c in r] for r in rows]
        widths = [max(len(r[i]) for r in cells) for i in range(len(header))]
        for j, r in enumerate(cells):
            self(" | ".join(c.ljust(w) for c, w in zip(r, widths)))
            if j == 0:
                self("-+-".join("-" * w for w in widths))

    def save(self) -> Path:
        RESULTS.mkdir(exist_ok=True)
        path = RESULTS / f"{self.name}.txt"
        path.write_text("\n".join(self.lines) + "\n")
        return path
