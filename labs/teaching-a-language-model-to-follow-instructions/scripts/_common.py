"""Shared helpers for the scripts: import path, hardware line, result files."""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
LAB_NAME = "teaching-a-language-model-to-follow-instructions"  # the directory this lab is filed under
RESULTS = LAB / "results"
sys.path.insert(0, str(LAB))

import torch  # noqa: E402


def cpu_name() -> str:
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def hardware(device: str | None = None) -> str:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        return (f"{torch.cuda.get_device_name(0)}, torch {torch.__version__.split('+')[0]}, "
                f"CUDA {torch.version.cuda}, bf16 autocast for training")
    return f"{cpu_name()}, {torch.get_num_threads()} threads, torch {torch.__version__.split('+')[0]}, CPU only"


def command() -> str:
    return "python " + " ".join([f"labs/{LAB_NAME}/scripts/{Path(sys.argv[0]).name}", *sys.argv[1:]])


class Report:
    """Collects printed lines and writes them to results/<name>.txt with a hardware and command header."""

    def __init__(self, name: str, device: str | None = None):
        self.path = RESULTS / f"{name}.txt"
        self.lines = [f"# hardware: {hardware(device)}", f"# command: {command()}", ""]

    def __call__(self, *parts) -> None:
        s = " ".join(str(p) for p in parts)
        print(s, flush=True)
        self.lines.append(s)

    def save(self) -> None:
        RESULTS.mkdir(exist_ok=True)
        self.path.write_text("\n".join(self.lines) + "\n")
        print(f"wrote {self.path.relative_to(LAB)}")


def save_json(name: str, obj, device: str | None = None) -> None:
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{name}.json").write_text(json.dumps({"hardware": hardware(device), "command": command(), **obj}, indent=1))
    print(f"wrote results/{name}.json")
