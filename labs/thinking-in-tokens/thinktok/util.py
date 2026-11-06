"""Hardware labels for result files. Only the CPU model, thread count and GPU name: never a host name."""

from __future__ import annotations

import torch


def cpu_name() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip().replace(" Processor", "")
    except OSError:
        pass
    import platform

    return platform.processor() or platform.machine()


def hardware(threads: int | None = None) -> str:
    if torch.cuda.is_available():
        return f"{torch.cuda.get_device_name(0)}, torch {torch.__version__}, CUDA {torch.version.cuda}"
    t = threads if threads is not None else torch.get_num_threads()
    return f"{cpu_name()}, {t} threads, torch {torch.__version__}"
