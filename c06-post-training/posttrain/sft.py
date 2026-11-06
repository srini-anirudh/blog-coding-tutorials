"""Stage 1: supervised fine-tuning with assistant-only labels, padded or packed batches, full FT or LoRA."""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from .chat import IGNORE_INDEX, ChatExample, encode_chat
from .packing import collate_packed, pack_examples, pad_batch
from .tasks import Problem, to_messages
from .utils import device_memory_mb, synchronize


def masked_lm_loss(logits: torch.Tensor, labels: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """Next-token cross-entropy on positions whose label is not -100.

    logits: (B, L, V), labels: (B, L). reduction="mean" divides by the number of target tokens in the batch,
    which is what Hugging Face does; reduction="sum" returns the total, for gradient accumulation.
    """
    shift_logits = logits[:, :-1].float()  # position t predicts token t+1
    shift_labels = labels[:, 1:]
    loss = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]),
        shift_labels.reshape(-1),
        ignore_index=IGNORE_INDEX,
        reduction="sum",
    )
    return loss / (shift_labels != IGNORE_INDEX).sum().clamp(min=1) if reduction == "mean" else loss


def build_examples(tok, problems: list[Problem], mode: str = "assistant") -> list[ChatExample]:
    return [encode_chat(tok, to_messages(p), mode=mode) for p in problems]


def batches(examples: list[ChatExample], batch_size: int, pad_id: int, pack: bool, max_len: int, seed: int):
    """Yield training batches forever. Packing builds rows of `max_len` tokens; padding uses `batch_size` rows."""
    rng = random.Random(seed)
    while True:
        order = examples[:]
        rng.shuffle(order)
        if pack:
            rows = pack_examples(order, max_len)
            rng.shuffle(rows)
            for s in range(0, len(rows) - batch_size + 1, batch_size):
                yield collate_packed(rows[s : s + batch_size], max_len, pad_id)
        else:
            for s in range(0, len(order) - batch_size + 1, batch_size):
                yield pad_batch(order[s : s + batch_size], pad_id)


@dataclass
class SFTConfig:
    steps: int = 200
    batch_size: int = 16
    lr: float = 5e-5
    warmup: int = 20
    weight_decay: float = 0.0
    pack: bool = False
    max_len: int = 256
    seed: int = 0
    log_every: int = 10
    eval_fn: object = None  # optional callable(model) -> dict, run every `eval_every` steps
    eval_every: int = 0
    history: list = field(default_factory=list)


def lr_at(step: int, cfg: SFTConfig) -> float:
    """Linear warmup, then cosine decay to 10% of the peak."""
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / cfg.warmup
    t = (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup)
    return cfg.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t)))


def train_sft(model, examples: list[ChatExample], pad_id: int, cfg: SFTConfig, device: torch.device) -> list[dict]:
    """Plain PyTorch SFT loop. Returns one log dict per `log_every` steps."""
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    model.train()
    stream = batches(examples, cfg.batch_size, pad_id, cfg.pack, cfg.max_len, cfg.seed)
    t0, tokens, peak = time.time(), 0, 0.0
    for step in range(cfg.steps):
        batch = {k: v.to(device) for k, v in next(stream).items()}
        for g in opt.param_groups:
            g["lr"] = lr_at(step, cfg)
        model_inputs = {k: batch[k] for k in ("input_ids", "attention_mask", "position_ids") if k in batch}
        logits = model(**model_inputs, use_cache=False).logits
        loss = masked_lm_loss(logits, batch["labels"])
        peak = max(peak, device_memory_mb(device))  # params + optimizer state + live activations
        loss.backward()
        peak = max(peak, device_memory_mb(device))  # params + optimizer state + gradients
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        tokens += int((batch["labels"] != IGNORE_INDEX).sum())
        if (step + 1) % cfg.log_every == 0 or step == 0:
            synchronize(device)
            row = {"step": step + 1, "loss": round(loss.item(), 4), "lr": lr_at(step, cfg),
                   "target_tokens": tokens, "sec": round(time.time() - t0, 1), "mem_mb": round(peak, 1)}
            if cfg.eval_fn is not None and cfg.eval_every and (step + 1) % cfg.eval_every == 0:
                row["eval"] = cfg.eval_fn(model)
            cfg.history.append(row)
            print(row, flush=True)
    return cfg.history
