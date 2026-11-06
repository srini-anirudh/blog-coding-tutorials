"""Every results file starts with the hardware it ran on and the command that wrote it."""

import os
import platform
import sys
from pathlib import Path

import torch

RESULTS = Path(__file__).resolve().parent.parent / "results"


def cpu_name():
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def hardware(device="cpu"):
    if str(device).startswith("cuda"):
        return (f"{torch.cuda.get_device_name(0)}, torch {torch.__version__}, CUDA {torch.version.cuda}")
    return f"{cpu_name()}, {torch.get_num_threads()} threads, torch {torch.__version__}"


class Report:
    """print() to the terminal and to results/<name>, with a two-line header."""

    def __init__(self, name, device="cpu"):
        RESULTS.mkdir(exist_ok=True)
        self.path = RESULTS / name
        self.f = open(self.path, "w")
        args = " ".join(sys.argv[1:])
        script = Path(sys.argv[0]).name
        self(f"# hardware: {hardware(device)}")
        self(f"# command: python labs/across-the-cnnverse/scripts/{script} {args}".rstrip())

    def __call__(self, *a):
        line = " ".join(str(x) for x in a)
        print(line)
        self.f.write(line + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


def set_threads():
    """Use OMP_NUM_THREADS if set, else 8."""
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 8)))
