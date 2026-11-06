"""Write each script's output to the terminal and to results/<name>.txt, headed by the hardware and the command."""

from __future__ import annotations

import json
import os
import platform
import sys
from pathlib import Path

import torch

LAB = Path(__file__).resolve().parents[1]
RESULTS = LAB / "results"
REPO = LAB.parents[1]


def hardware() -> str:
    if torch.cuda.is_available():
        return f"{torch.cuda.get_device_name(0)}, torch {torch.__version__}, CUDA {torch.version.cuda}"
    name = platform.processor() or "CPU"
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                name = line.split(":", 1)[1].strip().replace(" Processor", "")
                break
    except OSError:
        pass
    return f"{name}, {torch.get_num_threads()} threads, torch {torch.__version__}"


class Report:
    """Use as `with Report("01_axis_atlas") as say: say("...")`."""

    def __init__(self, name: str):
        self.name, self.lines = name, []
        rel = Path(sys.argv[0]).resolve().relative_to(REPO) if Path(sys.argv[0]).resolve().is_relative_to(REPO) else Path(sys.argv[0]).name
        args = " ".join(sys.argv[1:])
        self.header = [f"# {hardware()}", f"# python {rel} {args}".rstrip()]

    def __enter__(self):
        for h in self.header:
            print(h)
        return self

    def __call__(self, *parts) -> None:
        line = " ".join(str(p) for p in parts)
        print(line, flush=True)
        self.lines.append(line)

    def json(self, data: dict) -> None:
        if os.environ.get("NORMGEO_NO_WRITE"):
            return
        RESULTS.mkdir(exist_ok=True)
        payload = {"hardware": hardware(), "command": self.header[1][2:], **data}
        (RESULTS / f"{self.name}.json").write_text(json.dumps(payload, indent=1) + "\n")

    def __exit__(self, *exc):
        if exc[0] is None and not os.environ.get("NORMGEO_NO_WRITE"):
            RESULTS.mkdir(exist_ok=True)
            (RESULTS / f"{self.name}.txt").write_text("\n".join(self.header + self.lines) + "\n")
        return False


def shakespeare() -> str:
    path = REPO / "c04-llm" / "data" / "tinyshakespeare.txt"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing. It ships with the c04-llm tutorial in this repo.")
    return path.read_text()
