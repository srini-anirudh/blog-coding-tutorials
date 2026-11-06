"""Text statistics for generated token sequences, plus a hardware line for result files."""

from __future__ import annotations

import os
import platform
import subprocess
import sys

import torch


def repeated_ngram_fraction(tokens: list[int], n: int = 4) -> float:
    """Fraction of the n-grams in one sequence that already appeared earlier in it. 0 means no repeats."""
    grams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    if not grams:
        return 0.0
    seen, rep = set(), 0
    for gr in grams:
        rep += gr in seen
        seen.add(gr)
    return rep / len(grams)


def distinct_n(seqs: list[list[int]], n: int = 2) -> float:
    """Unique n-grams / total n-grams, pooled over several sequences."""
    grams = [tuple(s[i:i + n]) for s in seqs for i in range(len(s) - n + 1)]
    return len(set(grams)) / max(len(grams), 1)


def hardware_line() -> str:
    cpu = platform.processor() or "cpu"
    try:
        out = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
        cpu = next(l.split(":", 1)[1].strip() for l in out.splitlines() if l.startswith("Model name"))
    except Exception:
        pass
    import transformers

    versions = f"PyTorch {torch.__version__.split('+')[0]}, transformers {transformers.__version__}"
    cpu = f"{cpu.replace(' Processor', '')}, {torch.get_num_threads()} threads"
    if os.environ.get("NEXTTOKEN_DEVICE", "cpu").startswith("cuda") and torch.cuda.is_available():
        return (f"# {torch.cuda.get_device_name()} for model forward passes, CUDA {torch.version.cuda}; "
                f"decoding logic on CPU ({cpu}); {versions}")
    return f"# {cpu}, CPU only, {versions}"


def result_header(script: str) -> str:
    args = " ".join(sys.argv[1:])
    return f"{hardware_line()}\n$ python labs/choosing-the-next-token/scripts/{os.path.basename(script)} {args}".rstrip()
