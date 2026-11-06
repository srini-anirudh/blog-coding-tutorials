"""Small helpers shared by every script: one attention call, hardware labels, result files."""

from __future__ import annotations

import math
import platform
import re
import sys
from pathlib import Path

import torch

LAB = Path(__file__).resolve().parent.parent  # labs/how-attention-moves-information
REPO = LAB.parent.parent
C03 = REPO / "c03-attention"  # the attention tutorial, reused where it already builds a mechanism


def attention(q, k, v, mask=None, scale=None):
    """softmax(q k^T * scale + mask) v, with every step visible. mask: bool, True = may attend.

    Returns (output, weights). q (..., Nq, d), k (..., Nk, d), v (..., Nk, dv).
    """
    scale = 1.0 / math.sqrt(q.size(-1)) if scale is None else scale
    scores = q @ k.transpose(-2, -1) * scale
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return weights @ v, weights


def use_c03() -> None:
    """Make `import attnzoo` (the C3 package) work from this lab."""
    if str(C03) not in sys.path:
        sys.path.insert(0, str(C03))


def cpu_name() -> str:
    """The CPU model from the OS, never the machine's name."""
    try:
        with open("/proc/cpuinfo") as f:
            match = re.search(r"model name\s*:\s*(.+)", f.read())
    except OSError:
        match = None
    return match.group(1).replace(" Processor", "").strip() if match else platform.machine()


def hardware_note(device: str = "cpu") -> str:
    if device == "cuda":
        return f"{torch.cuda.get_device_name(0)}, torch {torch.__version__}, CUDA {torch.version.cuda}"
    return f"{cpu_name()}, {torch.get_num_threads()} threads, torch {torch.__version__}, CPU only"


def command_line() -> str:
    """The command that produced a result, written relative to the repository root."""
    script = Path(sys.argv[0]).resolve()
    try:
        script = script.relative_to(REPO)
    except ValueError:
        script = Path(script.name)
    return " ".join(["python", str(script), *sys.argv[1:]])


class Report:
    """Print lines and save them to results/<name>.txt, headed by the hardware and the command."""

    def __init__(self, name: str, device: str = "cpu") -> None:
        self.path = LAB / "results" / f"{name}.txt"
        self.lines = [f"# hardware: {hardware_note(device)}", f"# command: {command_line()}", ""]
        for line in self.lines[:2]:
            print(line)

    def __call__(self, line: str = "") -> None:
        print(line)
        self.lines.append(line)

    def table(self, header: list[str], rows: list[list]) -> None:
        cells = [header] + [[_fmt(c) for c in r] for r in rows]
        widths = [max(len(r[i]) for r in cells) for i in range(len(header))]
        for j, r in enumerate(cells):
            self("  ".join(c.ljust(w) if i == 0 else c.rjust(w) for i, (c, w) in enumerate(zip(r, widths))).rstrip())
            if j == 0:
                self("  ".join("-" * w for w in widths))

    def save(self) -> None:
        self.path.parent.mkdir(exist_ok=True)
        self.path.write_text("\n".join(self.lines) + "\n")
        print(f"\nsaved {self.path.relative_to(REPO)}")


def _fmt(c) -> str:
    if isinstance(c, float):
        if c == 0 or (1e-3 <= abs(c) < 1e5):
            return f"{c:.3f}"
        return f"{c:.2e}"
    return str(c)


def polite(threads: int = 8) -> None:
    torch.set_num_threads(min(threads, torch.get_num_threads()))
