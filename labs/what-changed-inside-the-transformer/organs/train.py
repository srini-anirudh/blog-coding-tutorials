"""A byte-level language-model training loop, small enough for a CPU, for comparing block variants."""

from __future__ import annotations

import math
import time
import urllib.request
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from ._paths import LAB, REPO
from .block import BlockConfig, Decoder

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def load_text() -> bytes:
    """Tiny Shakespeare, reused from C4's data folder if it is there, otherwise downloaded once into data/."""
    for path in (LAB / "data" / "tinyshakespeare.txt", REPO / "c04-llm" / "data" / "tinyshakespeare.txt"):
        if path.exists():
            return path.read_bytes()
    path = LAB / "data" / "tinyshakespeare.txt"
    path.parent.mkdir(exist_ok=True)
    urllib.request.urlretrieve(DATA_URL, path)
    return path.read_bytes()


@dataclass
class TrainConfig:
    steps: int = 300
    batch: int = 16
    seq: int = 128
    lr: float = 1e-3
    warmup: int = 0  # linear warmup steps; 0 means full lr from step 1
    clip: float | None = None
    seed: int = 0
    eval_every: int = 50


@dataclass
class TrainLog:
    losses: list[float] = field(default_factory=list)
    val: list[tuple[int, float]] = field(default_factory=list)
    first_layer_grad: list[float] = field(default_factory=list)
    seconds: float = 0.0

    def final_val(self) -> float:
        return self.val[-1][1] if self.val else float("nan")

    def spikes(self, ratio: float = 1.5) -> int:
        """Steps whose loss jumps to more than `ratio` times the median of the 20 steps before it."""
        n = 0
        for i in range(20, len(self.losses)):
            prev = sorted(self.losses[i - 20 : i])[10]
            n += self.losses[i] > ratio * prev
        return n


def batches(data: torch.Tensor, batch: int, seq: int, gen: torch.Generator):
    while True:
        idx = torch.randint(0, len(data) - seq - 1, (batch,), generator=gen)
        x = torch.stack([data[i : i + seq] for i in idx.tolist()])
        y = torch.stack([data[i + 1 : i + seq + 1] for i in idx.tolist()])
        yield x.long(), y.long()


def train(model_cfg: BlockConfig, cfg: TrainConfig, text: bytes | None = None) -> tuple[Decoder, TrainLog]:
    torch.manual_seed(cfg.seed)
    data = torch.frombuffer(bytearray(text if text is not None else load_text()), dtype=torch.uint8)
    split = int(0.9 * len(data))
    train_data, val_data = data[:split], data[split:]
    model = Decoder(model_cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.98), weight_decay=0.0)
    gen = torch.Generator().manual_seed(cfg.seed)
    val_gen = torch.Generator().manual_seed(1234)
    stream = batches(train_data, cfg.batch, cfg.seq, gen)
    first = model.blocks[0].attn.qkv.weight
    log = TrainLog()
    t0 = time.perf_counter()
    for step in range(1, cfg.steps + 1):
        lr = cfg.lr * min(1.0, step / cfg.warmup) if cfg.warmup else cfg.lr
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = next(stream)
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        log.first_layer_grad.append(first.grad.norm().item())
        if cfg.clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
        opt.step()
        log.losses.append(loss.item() if math.isfinite(loss.item()) else float("inf"))
        if step % cfg.eval_every == 0 or step == cfg.steps:
            log.val.append((step, evaluate(model, val_data, cfg, val_gen)))
    log.seconds = time.perf_counter() - t0
    return model, log


@torch.no_grad()
def evaluate(model: Decoder, data: torch.Tensor, cfg: TrainConfig, gen: torch.Generator, n: int = 4) -> float:
    model.eval()
    stream = batches(data, cfg.batch, cfg.seq, gen)
    total = 0.0
    for _ in range(n):
        x, y = next(stream)
        total += F.cross_entropy(model(x).flatten(0, 1), y.flatten()).item()
    model.train()
    return total / n
