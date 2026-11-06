"""Hardware line and small helpers shared by the scripts."""
from __future__ import annotations

import json
import os
import platform
from pathlib import Path

import torch


def hardware() -> str:
    if torch.cuda.is_available():
        return f"{torch.cuda.get_device_name(0)}, torch {torch.__version__.split('+')[0]}, CUDA {torch.version.cuda}"
    cpu = "CPU"
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        cpu = platform.processor() or "CPU"
    return f"{cpu}, {torch.get_num_threads()} threads, torch {torch.__version__.split('+')[0]}"


def header(cmd: str) -> str:
    return f"# hardware: {hardware()}\n# command: {cmd}"


def write(path: Path | str, cmd: str, lines: list[str], data: dict | None = None):
    """Write results/<name>.txt (header + lines) and, if given, results/<name>.json."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = header(cmd) + "\n\n" + "\n".join(lines) + "\n"
    path.with_suffix(".txt").write_text(text)
    if data is not None:
        data = {"hardware": hardware(), "command": cmd, **data}
        path.with_suffix(".json").write_text(json.dumps(data, indent=1))
    print(text)


def n_from_env(default: int) -> int:
    return int(os.environ.get("LAB_N", default))
