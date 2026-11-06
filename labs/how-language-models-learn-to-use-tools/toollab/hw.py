"""One line naming the hardware, and the command, for the top of every results file."""

from __future__ import annotations

import sys

import torch


def hardware() -> str:
    if torch.cuda.is_available():
        return f"{torch.cuda.get_device_name(0)}, torch {torch.__version__}, CUDA {torch.version.cuda}"
    cpu = "unknown CPU"
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return f"{cpu}, {torch.get_num_threads()} threads, torch {torch.__version__}"


def command() -> str:
    script = sys.argv[0]
    i = script.find("labs/")
    return " ".join(["python", script[i:] if i >= 0 else script, *sys.argv[1:]])


def header() -> dict:
    print(f"hardware: {hardware()}")
    print(f"command: {command()}\n", flush=True)
    return {"hardware": hardware(), "command": command()}
