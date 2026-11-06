"""Write a script's output to the terminal and to results/, with a hardware line and the command first."""

from __future__ import annotations

import json
import platform
from pathlib import Path

import torch


def hardware() -> str:
    if torch.cuda.is_available():
        return (f"{torch.cuda.get_device_name(0)}, torch {torch.__version__.split('+')[0]}, "
                f"CUDA {torch.version.cuda}")
    model = platform.processor() or "unknown CPU"
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    model = model.replace(" 48-Core Processor", "").replace(" 96-Core Processor", "")
    return f"{model}, {torch.get_num_threads()} threads, torch {torch.__version__.split('+')[0]}"


class Report:
    def __init__(self, out_stem: str | None, command: str):
        self.out = Path(out_stem) if out_stem else None
        self.lines = [f"# hardware: {hardware()}", f"# command: python labs/from-vlms-to-vlas/{command}", ""]
        for line in self.lines[:2]:
            print(line)

    def line(self, text: str = ""):
        print(text)
        self.lines.append(text)

    def table(self, rows: list[dict]):
        if not rows:
            return
        keys = list(rows[0])
        cells = [[str(r.get(k, "")) for k in keys] for r in rows]
        w = [max(len(k), *(len(c[i]) for c in cells)) for i, k in enumerate(keys)]
        self.line("  ".join(k.ljust(w[i]) for i, k in enumerate(keys)))
        for c in cells:
            self.line("  ".join(v.ljust(w[i]) for i, v in enumerate(c)))

    def save_json(self, obj):
        if self.out is None:
            return
        self.out.parent.mkdir(parents=True, exist_ok=True)
        self.out.with_suffix(".txt").write_text("\n".join(self.lines) + "\n")
        payload = {"hardware": hardware(), "command": self.lines[1][len("# command: "):], "results": obj}
        self.out.with_suffix(".json").write_text(json.dumps(payload, indent=1) + "\n")
