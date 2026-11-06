"""Small helpers shared by every module: device choice, hardware labels, tiny configs, parameter counts."""

from __future__ import annotations

import os
import platform
import re
import subprocess
import warnings
from pathlib import Path

import torch
from datasets.utils.logging import disable_progress_bar as disable_datasets_progress_bar
from torch import nn
from transformers import LlamaConfig
from transformers.utils import logging as hf_logging

# The checkpoints this tutorial uses. All are small enough for a laptop.
BASE_LM = "HuggingFaceTB/SmolLM2-135M"
CHAT_LM = "HuggingFaceTB/SmolLM2-135M-Instruct"
SMALL_VLM = "HuggingFaceTB/SmolVLM-256M-Instruct"


def pick_device() -> torch.device:
    """Prefer CUDA, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def hardware_label(device: str | torch.device = "cpu") -> str:
    """Name the hardware that produced a timing, by model only: "AMD EPYC 7R13, 8 threads"."""
    if torch.device(device).type == "cuda":
        return torch.cuda.get_device_name(0)
    model = platform.processor() or "unknown CPU"
    if platform.system() == "Linux":
        match = re.search(r"model name\s*:\s*(.+)", Path("/proc/cpuinfo").read_text())
        model = match.group(1) if match else model
    elif platform.system() == "Darwin":
        model = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    model = re.sub(r"\s+(\d+-Core\s+)?Processor\s*$", "", model)
    return f"{model}, {torch.get_num_threads()} threads"


def be_polite(threads: int = 8) -> None:
    """Cap CPU threads and silence progress noise, so timings are comparable and the output stays readable."""
    torch.set_num_threads(min(threads, os.cpu_count() or threads))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()
    disable_datasets_progress_bar()
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)


def tiny_llama_config(vocab_size: int = 512, **overrides) -> LlamaConfig:
    """A Llama-architecture config small enough to train in seconds on a CPU.

    Same code path as SmolLM2-135M (both are `LlamaForCausalLM`), so anything verified here holds there.
    """
    kwargs = dict(
        vocab_size=vocab_size,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        tie_word_embeddings=True,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    kwargs.update(overrides)
    return LlamaConfig(**kwargs)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return (trainable, total) parameter counts. Tied weights are counted once."""
    seen: set[int] = set()
    trainable = total = 0
    for p in model.parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    return trainable, total
