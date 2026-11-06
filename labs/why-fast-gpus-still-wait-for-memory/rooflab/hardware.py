"""Published peak rates, as quoted in the post, and a one-line hardware label for every saved output.

These are vendor peaks (dense, no 2:4 sparsity). The lab measures its own ceilings on the GPU it runs on;
these numbers are only what the measurements are compared against.
"""

import os
import platform
from dataclasses import dataclass

TB = 1e12
TF = 1e12


@dataclass(frozen=True)
class Accel:
    name: str
    hbm: float  # bytes/s
    bf16: float  # FLOP/s, dense tensor core
    fp8: float | None
    fp32: float  # FLOP/s, CUDA cores with FMA

    def ridge(self, dtype: str = "bf16", beta: float | None = None) -> float:
        """Machine balance I* = pi / beta, in FLOP per byte."""
        pi = {"bf16": self.bf16, "fp8": self.fp8, "fp32": self.fp32}[dtype]
        return pi / (beta or self.hbm)


# Table 1 of the post.
ACCELS = {
    "A100 SXM 80GB": Accel("A100 SXM 80GB", 2.04 * TB, 312 * TF, None, 19.5 * TF),
    "H100 SXM5": Accel("H100 SXM5", 3.35 * TB, 989 * TF, 1979 * TF, 66.9 * TF),
    "H200 SXM": Accel("H200 SXM", 4.80 * TB, 989 * TF, 1979 * TF, 66.9 * TF),
    "B200 (spec)": Accel("B200 (spec)", 8.00 * TB, 2250 * TF, 4500 * TF, 80 * TF),
    "MI300X": Accel("MI300X", 5.30 * TB, 1307 * TF, 2615 * TF, 163 * TF),
    "L40S": Accel("L40S", 0.86 * TB, 362 * TF, 733 * TF, 91.6 * TF),
    "RTX 4090": Accel("RTX 4090", 1.01 * TB, 165 * TF, 330 * TF, 82.6 * TF),
}
H100 = ACCELS["H100 SXM5"]
H100_TF32 = 495 * TF  # the post's compute-ceiling stack

# Table 4: off-chip links, per GPU.
LINKS = {"HBM3": 3.35 * TB, "NVLink 4": 450e9, "PCIe Gen5 x16": 64e9, "400 Gb/s fabric": 50e9}


def be_polite(threads: int = 4) -> None:
    """Keep CPU work small on a shared machine."""
    import torch

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", threads)))


def hardware_note() -> str:
    """One line naming the hardware and software. No hostnames."""
    import torch

    if torch.cuda.is_available():
        return f"{torch.cuda.get_device_name(0)}, torch {torch.__version__}, CUDA {torch.version.cuda}"
    cpu = platform.machine()
    try:
        with open("/proc/cpuinfo") as f:
            cpu = next(line.split(":", 1)[1].strip() for line in f if line.startswith("model name"))
    except (OSError, StopIteration):
        pass
    return f"CPU only ({cpu}, {torch.get_num_threads()} threads), torch {torch.__version__}"
