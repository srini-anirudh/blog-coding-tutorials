"""Tensor statistics for the sparsity and outlier sections, plus a tiny results writer."""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import torch


def exact_zero_fraction(a: torch.Tensor) -> float:
    return (a == 0).float().mean().item()


def near_zero_fraction(a: torch.Tensor, eps: float) -> float:
    return (a.abs() <= eps).float().mean().item()


def kurtosis(a: torch.Tensor) -> float:
    """Excess kurtosis of all entries. A Gaussian gives 0. Heavy tails give large positive values."""
    a = a.double().flatten()
    c = a - a.mean()
    return ((c**4).mean() / (c**2).mean() ** 2 - 3.0).item()


def max_over_rms(a: torch.Tensor) -> float:
    a = a.double().flatten()
    return (a.abs().max() / a.pow(2).mean().sqrt()).item()


def int8_per_tensor_error(a: torch.Tensor) -> float:
    """Relative L2 error of symmetric per-tensor INT8 (absmax scaling), the scheme the post says outliers break."""
    a = a.float()
    scale = a.abs().max() / 127.0
    q = torch.clamp(torch.round(a / scale), -127, 127) * scale
    return ((q - a).norm() / a.norm()).item()


def tensor_report(a: torch.Tensor) -> dict:
    return {
        "exact_zero": exact_zero_fraction(a),
        "near_zero_1e-2_rms": near_zero_fraction(a, 1e-2 * a.float().pow(2).mean().sqrt().item()),
        "kurtosis": kurtosis(a),
        "max_over_rms": max_over_rms(a),
        "int8_rel_err": int8_per_tensor_error(a),
    }


def hardware_line(device: str = "cpu") -> str:
    """One line naming the hardware. No hostnames. The CPU name is read from /proc/cpuinfo when present."""
    if device.startswith("cuda") and torch.cuda.is_available():
        hw = f"{torch.cuda.get_device_name()}, CUDA {torch.version.cuda}"
    else:
        name = platform.processor() or "CPU"
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    name = line.split(":", 1)[1].strip()
                    break
        except OSError:
            pass
        hw = f"{name}, {torch.get_num_threads()} threads"
    return f"# hardware: {hw}, torch {torch.__version__}"


class Results:
    """Collect printed lines and a JSON dict; write both to results/<name>.txt and .json."""

    def __init__(self, name: str, out_dir: Path, device: str = "cpu") -> None:
        self.name, self.out_dir = name, out_dir
        cmd = " ".join([Path(sys.argv[0]).name] + sys.argv[1:])
        self.lines = [hardware_line(device), f"# command: python scripts/{cmd}"]
        self.data: dict = {}
        for line in self.lines:
            print(line)

    def print(self, *parts) -> None:
        line = " ".join(str(p) for p in parts)
        print(line, flush=True)
        self.lines.append(line)

    def save(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / f"{self.name}.txt").write_text("\n".join(self.lines) + "\n")
        payload = {"hardware": self.lines[0][2:], "command": self.lines[1][2:], **self.data}
        (self.out_dir / f"{self.name}.json").write_text(json.dumps(payload, indent=2) + "\n")
