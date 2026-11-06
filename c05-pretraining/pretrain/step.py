"""One optimiser step: gradient accumulation, autocast, clipping, and a finite check.

    B_global = B_micro * N_accum * world_size      (sequences per optimiser step)

The loss of every micro-batch is a *sum* over its tokens divided by the number of target
tokens in the *whole* step. Summing those backward passes gives exactly the gradient of
the mean loss over all the step's tokens, which is what one big batch would give.
Dividing each micro-batch by its own token count instead (mean of means) is only equal
when every micro-batch has the same number of target tokens.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from pretrain.packing import IGNORE


@dataclass
class StepResult:
    loss: float  # mean loss per target token over the whole step
    grad_norm: float  # total L2 norm of the gradient before clipping
    tokens: int  # target tokens in the step (this process)
    skipped: bool  # True when the gradient was not finite and no update was applied


def lm_loss_sum(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """logits (B, L, V), targets (B, L) -> summed cross-entropy over non-ignored targets.

    Logits are upcast to fp32 first: softmax over V entries in bf16 loses precision.
    """
    return F.cross_entropy(
        logits.float().reshape(-1, logits.size(-1)), targets.reshape(-1),
        ignore_index=IGNORE, reduction="sum",
    )


def autocast(device: torch.device, precision: str):
    if precision == "fp32":
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16)


def train_step(
    model: nn.Module,
    opt: torch.optim.Optimizer,
    sched: torch.optim.lr_scheduler.LRScheduler | None,
    micro_batches: list[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    precision: str = "bf16",
    grad_clip: float = 1.0,
) -> StepResult:
    """Run len(micro_batches) forward/backward passes and one optimiser step."""
    n_tokens = sum(int((y != IGNORE).sum()) for _, y in micro_batches)
    loss_total = 0.0
    for i, (x, y) in enumerate(micro_batches):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        last = i == len(micro_batches) - 1
        # Under DDP, skip the gradient all-reduce on every micro-batch but the last.
        sync = contextlib.nullcontext() if last or not hasattr(model, "no_sync") else model.no_sync()
        with sync:
            with autocast(device, precision):
                logits = model(x).logits  # (B, L, V)
            loss_sum = lm_loss_sum(logits, y)
            (loss_sum / n_tokens).backward()
        loss_total += loss_sum.item()

    # clip_grad_norm_ returns the norm *before* clipping: the number worth logging.
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip).item()
    skipped = not torch.isfinite(torch.tensor(grad_norm))
    if not skipped:
        opt.step()
    opt.zero_grad(set_to_none=True)
    if sched is not None:
        sched.step()  # advance even on a skipped step, so the schedule stays aligned with step
    return StepResult(loss_total / n_tokens, grad_norm, n_tokens, skipped)
