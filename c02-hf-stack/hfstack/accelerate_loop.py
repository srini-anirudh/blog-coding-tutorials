"""The raw loop again, with Accelerate handling devices and processes.

Accelerate does not own the loop. It wraps the model, optimizer and dataloader so the same code runs on
one CPU, one GPU, or N processes under `torchrun` / `accelerate launch`:

* `prepare` moves the model to the device, wraps it in DDP when there are several processes,
  and gives each process its own shard of the dataloader.
* `backward` applies mixed-precision loss scaling when it is enabled.
* `no_sync` skips the gradient all-reduce on micro-batches that are not the last of an accumulation.
"""

from __future__ import annotations

from contextlib import nullcontext

import torch
from accelerate import Accelerator
from torch import nn
from torch.utils.data import DataLoader, SequentialSampler
from transformers import get_scheduler

from .training import LoopConfig, count_label_tokens, make_optimizer, summed_token_loss


def train_accelerate(
    model: nn.Module, dataset, collator, cfg: LoopConfig, accelerator: Accelerator | None = None
) -> dict[str, list[float]]:
    """Same choices as `training.train_raw`; returns the same per-step history on every process."""
    accelerator = accelerator or Accelerator(cpu=True)
    torch.manual_seed(cfg.seed)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, sampler=SequentialSampler(dataset), collate_fn=collator)
    optimizer = make_optimizer(model, cfg)
    scheduler = get_scheduler(
        "linear", optimizer, num_warmup_steps=cfg.warmup_steps, num_training_steps=cfg.max_steps
    )
    # The scheduler is left unprepared on purpose: a prepared scheduler steps once per process per
    # optimizer step, which changes the schedule when the process count changes.
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)

    history: dict[str, list[float]] = {"loss": [], "grad_norm": [], "lr": []}
    model.train()
    batches = iter(loader)
    for step in range(cfg.max_steps):
        micro_batches = []
        for _ in range(cfg.grad_accum):
            try:
                micro_batches.append(next(batches))
            except StopIteration:
                batches = iter(loader)
                micro_batches.append(next(batches))

        # Count label tokens on every process and sum them, so each token in the global batch has equal
        # weight. DDP averages gradients over processes, hence the factor num_processes.
        local_tokens = torch.tensor(sum(count_label_tokens(b["labels"]) for b in micro_batches))
        n_tokens = accelerator.reduce(local_tokens.to(accelerator.device), reduction="sum").item()
        scale = accelerator.num_processes / n_tokens

        step_loss = torch.zeros((), device=accelerator.device)
        for i, batch in enumerate(micro_batches):
            last = i == len(micro_batches) - 1
            # Only the last micro-batch triggers the all-reduce of gradients.
            with accelerator.no_sync(model) if not last else nullcontext():
                logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
                loss = summed_token_loss(logits, batch["labels"]) * scale
                accelerator.backward(loss)
            step_loss += loss.detach() / accelerator.num_processes

        grad_norm = accelerator.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        history["lr"].append(scheduler.get_last_lr()[0])
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        history["loss"].append(accelerator.reduce(step_loss, reduction="sum").item())
        history["grad_norm"].append(float(grad_norm))
    return history

