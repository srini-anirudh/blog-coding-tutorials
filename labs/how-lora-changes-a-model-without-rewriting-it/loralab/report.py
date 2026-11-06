"""Write results files that start with the hardware and the command that produced them."""

from __future__ import annotations

import json
import os
import platform
import sys
from pathlib import Path

import torch

RESULTS = Path(__file__).resolve().parent.parent / "results"


def hardware(device: str = "cpu") -> str:
    if device.startswith("cuda") and torch.cuda.is_available():
        return f"{torch.cuda.get_device_name(0)}, torch {torch.__version__}, CUDA {torch.version.cuda}"
    cpu = platform.processor() or platform.machine()
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip().replace(" Processor", "")
                break
    except OSError:
        pass
    return f"{cpu}, {torch.get_num_threads()} threads, torch {torch.__version__}"


def command() -> str:
    script = Path(sys.argv[0])
    rel = script.name if script.parent.name != "scripts" else f"scripts/{script.name}"
    return " ".join(["python", rel, *sys.argv[1:]])


class Report:
    """Collect printed lines and a JSON payload, then save both under results/<name>.{txt,json}."""

    def __init__(self, name: str, device: str = "cpu"):
        self.name, self.lines, self.data = name, [], {}
        self.header = [f"# hardware: {hardware(device)}", f"# command: {command()}"]
        for h in self.header:
            print(h)

    def __call__(self, *parts) -> None:
        line = " ".join(str(p) for p in parts)
        print(line, flush=True)
        self.lines.append(line)

    def save(self) -> None:
        RESULTS.mkdir(exist_ok=True)
        (RESULTS / f"{self.name}.txt").write_text("\n".join(self.header + self.lines) + "\n")
        payload = {"hardware": self.header[0][12:], "command": self.header[1][11:], **self.data}
        (RESULTS / f"{self.name}.json").write_text(json.dumps(payload, indent=1, default=float) + "\n")


def table(rows: list[list], header: list[str]) -> list[str]:
    cells = [header] + [[f"{c:.4g}" if isinstance(c, float) else str(c) for c in r] for r in rows]
    widths = [max(len(r[i]) for r in cells) for i in range(len(header))]
    fmt = lambda r: "  ".join(c.rjust(w) for c, w in zip(r, widths))
    return [fmt(cells[0]), fmt(["-" * w for w in widths])] + [fmt(r) for r in cells[1:]]


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
