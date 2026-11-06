"""Write a results file that starts with the hardware line and the command, as the lab requires."""

from __future__ import annotations

import json
import os
import platform
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"


def hardware() -> str:
    import torch

    if torch.cuda.is_available():
        return (f"hardware: {torch.cuda.get_device_name(0)}, torch {torch.__version__}, "
                f"CUDA {torch.version.cuda}")
    cpu = platform.processor() or "CPU"
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return f"hardware: {cpu}, {torch.get_num_threads()} threads, torch {torch.__version__}"


def command() -> str:
    script = Path(sys.argv[0])
    rel = Path("labs") / script.resolve().parent.parent.name / script.parent.name / script.name
    return "command: python " + " ".join([str(rel)] + sys.argv[1:])


class Report:
    """Collect printed lines, then save them to results/<name>.txt (and optional JSON)."""

    def __init__(self, name: str):
        self.name = name
        self.lines = [hardware(), command(), ""]
        for line in self.lines[:2]:
            print(line)
        print()

    def __call__(self, *parts):
        line = " ".join(str(p) for p in parts)
        print(line)
        self.lines.append(line)

    def save(self, data: dict | None = None):
        RESULTS.mkdir(exist_ok=True)
        (RESULTS / f"{self.name}.txt").write_text("\n".join(self.lines) + "\n")
        if data is not None:
            payload = {"hardware": self.lines[0], "command": self.lines[1], **data}
            (RESULTS / f"{self.name}.json").write_text(json.dumps(payload, indent=1) + "\n")
        print(f"\nwrote results/{self.name}.txt", file=sys.stderr)


def pick_device(arg: str = "auto") -> str:
    import torch

    if arg != "auto":
        return arg
    return "cuda" if torch.cuda.is_available() else "cpu"


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
