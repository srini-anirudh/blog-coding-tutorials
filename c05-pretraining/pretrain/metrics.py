"""Training metrics: one JSON line per log step, plus a short console line.

JSONL is append-only and survives crashes: after a resume the file simply continues.
Every number the tutorial plots comes from these files.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn


@torch.no_grad()
def param_norm(model: nn.Module) -> float:
    """Global L2 norm of all parameters. Weight decay and the LR shape its trajectory."""
    return math.sqrt(sum(p.float().pow(2).sum().item() for p in model.parameters()))


class MetricsLogger:
    def __init__(self, out_dir: str | Path, rank: int = 0, filename: str = "metrics.jsonl"):
        self.rank = rank
        self.path = Path(out_dir) / filename
        if rank == 0:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.t0 = time.time()

    def log(self, record: dict, echo: bool = True) -> None:
        if self.rank != 0:
            return
        record = {"time": round(time.time() - self.t0, 2), **record}
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")
        if echo:
            print(" ".join(_fmt(k, v) for k, v in record.items()), flush=True)


def _fmt(k: str, v) -> str:
    if isinstance(v, float):
        return f"{k} {v:.3g}" if abs(v) < 1e-2 or abs(v) >= 1e4 else f"{k} {v:.4f}"
    return f"{k} {v}"


def read_metrics(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
