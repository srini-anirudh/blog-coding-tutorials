"""Write script output to results/ with a hardware line and the command on top."""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import torch

RESULTS = Path(__file__).resolve().parents[1] / "results"


def hardware(device: str = "cpu") -> str:
    if device.startswith("cuda"):
        name = torch.cuda.get_device_name(0)
        return f"{name}, torch {torch.__version__}, CUDA {torch.version.cuda}"
    cpu = platform.processor() or "unknown CPU"
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return f"{cpu}, {torch.get_num_threads()} threads, torch {torch.__version__}"


class Report:
    """Collects printed lines, then writes results/<name>.txt (and .json if data was added)."""

    def __init__(self, name: str, device: str = "cpu") -> None:
        self.name = name
        script = Path(sys.argv[0]).resolve()
        try:
            script = script.relative_to(RESULTS.parents[2])  # path from the repository root
        except ValueError:
            script = Path(script.name)
        self.lines = [f"# hardware: {hardware(device)}",
                      "# command: python " + " ".join([script.as_posix()] + sys.argv[1:])]
        self.data: dict = {"hardware": hardware(device)}
        print("\n".join(self.lines))

    def __call__(self, *parts) -> None:
        line = " ".join(str(p) for p in parts)
        print(line, flush=True)
        self.lines.append(line)

    def table(self, header: list[str], rows: list[list]) -> None:
        cells = [header] + [[f"{c:.4g}" if isinstance(c, float) else str(c) for c in r] for r in rows]
        widths = [max(len(r[i]) for r in cells) for i in range(len(header))]
        for j, r in enumerate(cells):
            self(" | ".join(c.ljust(w) for c, w in zip(r, widths)))
            if j == 0:
                self("-+-".join("-" * w for w in widths))

    def save(self) -> Path:
        RESULTS.mkdir(exist_ok=True)
        path = RESULTS / f"{self.name}.txt"
        path.write_text("\n".join(self.lines) + "\n")
        if len(self.data) > 1:
            (RESULTS / f"{self.name}.json").write_text(json.dumps(self.data, indent=1, default=str) + "\n")
        return path
