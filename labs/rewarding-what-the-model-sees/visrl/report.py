"""Write results files that start with the hardware and the command that produced them."""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import torch

LAB = Path(__file__).resolve().parents[1]
RESULTS = LAB / "results"
REPO = LAB.parents[1]


def cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def header() -> str:
    script = Path(sys.argv[0]).resolve()
    try:
        script = script.relative_to(REPO)
    except ValueError:
        script = Path(script.name)
    cmd = " ".join(["python", str(script), *sys.argv[1:]])
    return (f"# hardware: {cpu_name()}, {torch.get_num_threads()} threads, CPU only, torch {torch.__version__}\n"
            f"# command: {cmd}\n")


def write(name: str, text: str, data: dict | None = None) -> None:
    """results/<name>.txt with the header, and results/<name>.json if data is given."""
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{name}.txt").write_text(header() + text.rstrip() + "\n")
    if data is not None:
        payload = {"hardware": header().splitlines()[0][2:], "command": header().splitlines()[1][2:], **data}
        (RESULTS / f"{name}.json").write_text(json.dumps(payload, indent=1) + "\n")
    print(text)


def table(rows: list[list], head: list[str]) -> str:
    """A plain Markdown table."""
    cells = [[str(c) for c in r] for r in [head, *rows]]
    w = [max(len(r[i]) for r in cells) for i in range(len(head))]
    fmt = lambda r: "| " + " | ".join(c.ljust(w[i]) for i, c in enumerate(r)) + " |"   # noqa: E731
    return "\n".join([fmt(cells[0]), "|" + "|".join("-" * (x + 2) for x in w) + "|", *map(fmt, cells[1:])])
