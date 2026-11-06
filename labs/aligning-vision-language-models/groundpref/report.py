"""Write results files that start with the hardware and the command that produced them."""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import torch

RESULTS = Path(__file__).resolve().parents[1] / "results"


def hardware() -> str:
    cpu = platform.processor() or "unknown CPU"
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    hw = f"{cpu}, {torch.get_num_threads()} threads, torch {torch.__version__}"
    if torch.cuda.is_available():
        hw = f"{torch.cuda.get_device_name(0)}, torch {torch.__version__}, CUDA {torch.version.cuda}"
    return hw


def command() -> str:
    parts = [Path(sys.argv[0]).resolve().as_posix()]
    i = parts[0].find("labs/")
    parts[0] = parts[0][i:] if i >= 0 else Path(parts[0]).name
    return "python " + " ".join(parts + sys.argv[1:])


class Log:
    """Print and keep every line, then save it as results/<name>.txt (and .json if given)."""

    def __init__(self, name: str):
        self.name, self.lines = name, [f"# hardware: {hardware()}", f"# command: {command()}"]
        print("\n".join(self.lines))

    def __call__(self, *args) -> None:
        line = " ".join(str(a) for a in args)
        print(line, flush=True)
        self.lines.append(line)

    def save(self, data: dict | None = None) -> None:
        RESULTS.mkdir(exist_ok=True)
        (RESULTS / f"{self.name}.txt").write_text("\n".join(self.lines) + "\n")
        if data is not None:
            payload = {"hardware": hardware(), "command": command(), **data}
            (RESULTS / f"{self.name}.json").write_text(json.dumps(payload, indent=1) + "\n")
