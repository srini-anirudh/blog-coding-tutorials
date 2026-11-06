"""Hardware labels, timing and result files.

Every result file starts with one line naming the hardware and one line with the command that wrote it.
"""

import os
import platform
import statistics
import sys
import time
from pathlib import Path

import torch

LAB = Path(__file__).resolve().parents[1]
RESULTS = LAB / "results"

# Published H100 SXM5 figures used by the post (spec, dense, no sparsity).
H100_SPEC = {
    "bf16_tflops": 989.0,
    "fp32_tflops": 67.0,
    "hbm_tbps": 3.35,
    "sms": 132,
}


def cpu_name() -> str:
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def hardware_line(device: str) -> str:
    """One line naming the hardware, with no host, user or path in it."""
    if device == "cuda":
        name = torch.cuda.get_device_name()
        return f"hardware: {name}, torch {torch.__version__}, CUDA {torch.version.cuda}"
    return f"hardware: {cpu_name()}, {torch.get_num_threads()} threads, torch {torch.__version__}"


def command_line() -> str:
    """The command as run from the repository root, with no absolute path."""
    parts = []
    for a in sys.argv:
        p = Path(a)
        if p.is_absolute() and LAB in p.parents:
            a = str(Path("labs") / LAB.name / p.relative_to(LAB))
        parts.append(a)
    return "command: python " + " ".join(parts)


class Report:
    """Collects printed lines and writes them, with the header, to results/<name>."""

    def __init__(self, name: str, device: str):
        self.name, self.lines = name, [hardware_line(device), command_line(), ""]
        print("\n".join(self.lines))

    def __call__(self, text: str = "") -> None:
        print(text, flush=True)
        self.lines.append(text)

    def table(self, header: list[str], rows: list[list]) -> None:
        cells = [header] + [[_fmt(c) for c in r] for r in rows]
        widths = [max(len(r[i]) for r in cells) for i in range(len(header))]
        for j, r in enumerate(cells):
            self(" | ".join(c.rjust(w) for c, w in zip(r, widths)))
            if j == 0:
                self("-+-".join("-" * w for w in widths))

    def save(self) -> Path:
        RESULTS.mkdir(exist_ok=True)
        path = RESULTS / self.name
        path.write_text("\n".join(self.lines) + "\n")
        return path


def _fmt(c) -> str:
    if isinstance(c, float):
        if c == 0 or 0.01 <= abs(c) < 1e5:
            return f"{c:.3g}" if abs(c) < 10 else f"{c:,.1f}"
        return f"{c:.3g}"
    return str(c)


def device_from_argv(default: str = "cuda") -> str:
    if "--device" in sys.argv:
        return sys.argv[sys.argv.index("--device") + 1]
    return default if torch.cuda.is_available() else "cpu"


def bench(fn, device: str, reps: int = 10, trials: int = 5, warmup: int = 2) -> float:
    """Median seconds per call of fn. CUDA calls are timed with events, CPU calls with a wall clock."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(trials):
        if device == "cuda":
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start.record()
            for _ in range(reps):
                fn()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end) / 1e3 / reps)
        else:
            t0 = time.perf_counter()
            for _ in range(reps):
                fn()
            times.append((time.perf_counter() - t0) / reps)
    return statistics.median(times)


def set_cpu_threads() -> None:
    n = int(os.environ.get("OMP_NUM_THREADS", "8"))
    torch.set_num_threads(n)
