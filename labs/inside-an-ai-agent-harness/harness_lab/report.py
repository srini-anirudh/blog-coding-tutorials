"""Result files: every one starts with a hardware line and the command that wrote it."""

from __future__ import annotations

import platform
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results"


def hardware() -> str:
    """A hardware label with no host name: CPU model and threads, or the GPU name."""
    try:
        import torch
        tv = f"torch {torch.__version__}"
        if torch.cuda.is_available():
            return f"{torch.cuda.get_device_name(0)}, {tv}, CUDA {torch.version.cuda}"
        threads = torch.get_num_threads()
    except ImportError:  # pragma: no cover
        tv, threads = "no torch", None
    cpu = platform.processor() or "unknown CPU"
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip().replace(" Processor", "")
                break
    except OSError:
        pass
    return f"{cpu}, {threads} threads, CPU only, {tv}"


def command() -> str:
    """The command as typed from the repo root, with any absolute prefix removed."""
    script = Path(sys.argv[0]).resolve()
    try:
        rel = script.relative_to(RESULTS.parents[2])
    except ValueError:
        rel = Path(script.name)
    return " ".join(["python", str(rel), *sys.argv[1:]])


class Report:
    """Collect lines, print them, and write them to results/<name> with the header."""

    def __init__(self, name: str):
        self.path = RESULTS / name
        self.lines = [f"hardware: {hardware()}", f"command: {command()}", ""]

    def __call__(self, *parts) -> None:
        line = " ".join(str(p) for p in parts)
        print(line)
        self.lines.append(line)

    def save(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("\n".join(self.lines) + "\n")
        return self.path
