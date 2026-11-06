"""Timing helpers and the hardware line every results file starts with."""

from __future__ import annotations

import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import torch

RESULTS = Path(__file__).resolve().parent.parent / "results"


def hardware() -> str:
    torch_part = f"torch {torch.__version__}"
    if torch.cuda.is_available():
        return f"{torch.cuda.get_device_name(0)}, {torch_part}, CUDA {torch.version.cuda}"
    cpu = platform.processor() or platform.machine()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return f"{cpu}, {torch.get_num_threads()} threads, {torch_part}, no GPU"


def command() -> str:
    """The command that produced a results file, with the path relative to the lab folder."""
    script = Path(sys.argv[0])
    rel = script.name if script.parent.name != "scripts" else f"scripts/{script.name}"
    return " ".join(["python", rel, *sys.argv[1:]])


class Report:
    """Prints as it goes and writes results/<name>.txt (and .json) starting with the hardware line."""

    def __init__(self, name: str):
        self.name, self.lines, self.data = name, [f"# hardware: {hardware()}", f"# command: {command()}"], {}
        for line in self.lines:
            print(line)

    def __call__(self, *parts) -> None:
        line = " ".join(str(p) for p in parts)
        print(line, flush=True)
        self.lines.append(line)

    def save(self) -> Path:
        RESULTS.mkdir(exist_ok=True)
        out = RESULTS / f"{self.name}.txt"
        out.write_text("\n".join(self.lines) + "\n")
        if self.data:
            (RESULTS / f"{self.name}.json").write_text(json.dumps({"hardware": hardware(), "command": command(),
                                                                   **self.data}, indent=1, default=float))
        return out


def cuda_time(fn, iters: int = 20, warmup: int = 5) -> float:
    """Median seconds per call, measured with CUDA events so launch-queue depth does not hide anything."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b) / 1e3)
    return statistics.median(times)


def wall_time(fn, iters: int = 20, warmup: int = 3, sync=None) -> float:
    """Median wall seconds per call. Pass `sync` (e.g. torch.cuda.synchronize) when work is asynchronous."""
    for _ in range(warmup):
        fn()
    if sync:
        sync()
    times = []
    for _ in range(iters):
        t = time.perf_counter()
        fn()
        if sync:
            sync()
        times.append(time.perf_counter() - t)
    return statistics.median(times)


def need_cuda(report: Report) -> bool:
    if torch.cuda.is_available():
        return True
    report("No CUDA device. This script measures GPU behaviour; run it on a GPU. Nothing was measured.")
    return False


def threads_from_env(default: int = 4) -> None:
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", default)))


def nvml_utilization():
    """Return a function that reads utilization.gpu (percent) for CUDA device 0 through libnvidia-ml, the
    counter nvidia-smi prints. Uses ctypes so pynvml is not needed. Returns None if NVML is unavailable."""
    import ctypes

    try:
        lib = ctypes.CDLL("libnvidia-ml.so.1")
    except OSError:
        return None

    class Util(ctypes.Structure):
        _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]

    if lib.nvmlInit_v2() != 0:
        return None
    p = torch.cuda.get_device_properties(0)
    bus = f"{p.pci_domain_id:08x}:{p.pci_bus_id:02x}:{p.pci_device_id:02x}.0".encode()
    handle = ctypes.c_void_p()
    if lib.nvmlDeviceGetHandleByPciBusId_v2(bus, ctypes.byref(handle)) != 0:
        return None
    u = Util()

    def read() -> int:
        lib.nvmlDeviceGetUtilizationRates(handle, ctypes.byref(u))
        return int(u.gpu)

    return read
