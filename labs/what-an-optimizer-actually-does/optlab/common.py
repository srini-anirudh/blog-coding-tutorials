"""Shared helpers: hardware labels, result files, and the small language model used for training runs.

The model is C4's TinyLLM (c04-llm/tinyllm), a Llama-style decoder, imported rather than rebuilt.
The data is C4's copy of Tiny Shakespeare, read as bytes.
"""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import torch

LAB = Path(__file__).resolve().parents[1]
REPO = LAB.parents[1]
RESULTS = LAB / "results"
sys.path.insert(0, str(REPO / "c04-llm"))


def hardware() -> str:
    if torch.cuda.is_available():
        return f"{torch.cuda.get_device_name(0)}, torch {torch.__version__}, CUDA {torch.version.cuda}"
    cpu = platform.processor() or "CPU"
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip().replace(" Processor", "")
                break
    except OSError:
        pass
    return f"{cpu}, {torch.get_num_threads()} threads, torch {torch.__version__}"


def command() -> str:
    """The command that produced a result, with the script path relative to the repository root."""
    script = Path(sys.argv[0]).resolve()
    try:
        script = script.relative_to(REPO)
    except ValueError:
        script = Path(script.name)
    return " ".join(["python", str(script), *sys.argv[1:]])


class Report:
    """Print lines and also write them to results/<name>.txt, headed by hardware and command."""

    def __init__(self, name: str) -> None:
        self.path = RESULTS / f"{name}.txt"
        self.lines = [f"# hardware: {hardware()}", f"# command: {command()}"]
        print("\n".join(self.lines))

    def __call__(self, *parts) -> None:
        line = " ".join(str(p) for p in parts)
        print(line)
        self.lines.append(line)

    def save(self, data: dict | None = None) -> None:
        RESULTS.mkdir(exist_ok=True)
        self.path.write_text("\n".join(self.lines) + "\n")
        if data is not None:
            data = {"hardware": hardware(), "command": command(), **data}
            self.path.with_suffix(".json").write_text(json.dumps(data, indent=1) + "\n")


def tiny_model(hidden=128, layers=4, heads=4, kv_heads=2, ffn=352, seq=128, seed=0):
    from tinyllm import ModelConfig, TinyLLM

    torch.manual_seed(seed)
    cfg = ModelConfig(vocab_size=256, hidden_size=hidden, num_layers=layers, num_heads=heads,
                      num_kv_heads=kv_heads, intermediate_size=ffn, max_seq_len=seq)
    return TinyLLM(cfg)


def shakespeare_batches(batch: int, seq: int, seed: int = 0, split: str = "train"):
    """An endless stream of (B, seq) byte windows from Tiny Shakespeare. 90/10 train/val split.

    TinyLLM shifts labels inside its loss, so each window is both the input and the labels.
    """
    data = torch.frombuffer(bytearray((REPO / "c04-llm/data/tinyshakespeare.txt").read_bytes()), dtype=torch.uint8)
    cut = int(0.9 * len(data))
    data = (data[:cut] if split == "train" else data[cut:]).long()
    gen = torch.Generator().manual_seed(seed)
    while True:
        idx = torch.randint(len(data) - seq, (batch,), generator=gen)
        yield torch.stack([data[i : i + seq] for i in idx])


def split_for_muon(model) -> tuple[list, list]:
    """Hidden 2-D weights go to Muon. Embeddings (tied to the LM head), norms and biases stay on AdamW."""
    hidden, rest = [], []
    for name, p in model.named_parameters():
        (hidden if p.ndim == 2 and "embed" not in name and "lm_head" not in name else rest).append(p)
    return hidden, rest
