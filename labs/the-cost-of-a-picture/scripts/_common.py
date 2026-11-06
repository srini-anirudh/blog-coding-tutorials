"""Shared setup for the scripts: import path, hardware line, and writing results with a header."""

from __future__ import annotations

import json
import os
import platform
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))
RESULTS = LAB / "results"

import torch  # noqa: E402


def hardware(device: str = "cpu") -> str:
    if str(device).startswith("cuda"):
        return (f"{torch.cuda.get_device_name(0)}, torch {torch.__version__.split('+')[0]}, "
                f"CUDA {torch.version.cuda}")
    cpu = "unknown CPU"
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip().replace(" Processor", "")
                break
    except OSError:
        cpu = platform.processor() or cpu
    return f"{cpu}, {torch.get_num_threads()} threads, torch {torch.__version__.split('+')[0]}"


def command() -> str:
    script = Path(sys.argv[0]).resolve().relative_to(LAB.parents[1])
    return " ".join(["python", str(script), *sys.argv[1:]])


def write(name: str, text: str, data: dict | None = None, device: str = "cpu") -> None:
    """results/<name>.txt (and .json): first line the hardware, second line the command."""
    RESULTS.mkdir(exist_ok=True)
    head = f"# hardware: {hardware(device)}\n# command: {command()}\n"
    (RESULTS / f"{name}.txt").write_text(head + text.rstrip() + "\n")
    if data is not None:
        payload = {"hardware": hardware(device), "command": command(), **data}
        (RESULTS / f"{name}.json").write_text(json.dumps(payload, indent=1) + "\n")
    print(text)


def quiet_transformers() -> None:
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    import transformers
    transformers.logging.set_verbosity_error()
    transformers.logging.disable_progress_bar()
