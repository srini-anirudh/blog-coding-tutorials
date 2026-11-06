"""The same training run three ways: a raw PyTorch loop, `Trainer`, and (in accelerate_loop.py) Accelerate.

The raw loop is written to make exactly the choices `Trainer` makes by default, so the two produce the
same loss curve to float precision. Each choice is a line below:

* data order: sequential sampler, `batch_size` examples per micro-batch, no drop_last
* loss: token-level cross-entropy, summed, divided by the number of label tokens in the whole
  accumulated batch (not by the micro-batch mean)
* optimizer: AdamW with weight decay on matrices only (no decay on biases and norm weights)
* schedule: linear warmup, then linear decay to zero at `max_steps`
* gradient clipping at `max_grad_norm` before each optimizer step
"""

from __future__ import annotations

import copy
import tempfile
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, SequentialSampler
from transformers import PrinterCallback, Trainer, TrainingArguments, get_scheduler

IGNORE_INDEX = -100


@dataclass
class LoopConfig:
    learning_rate: float = 3e-3
    batch_size: int = 4
    grad_accum: int = 2
    max_steps: int = 20
    warmup_steps: int = 4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    seed: int = 0


def summed_token_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Sum of next-token cross-entropy over every label that is not -100.

    logits: (B, N, V). labels: (B, N), aligned with input_ids; position t is predicted from tokens < t.
    """
    shift_logits = logits[:, :-1, :].float()  # (B, N-1, V): predictions for positions 1..N-1
    shift_labels = labels[:, 1:]  # (B, N-1)
    return F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        ignore_index=IGNORE_INDEX,
        reduction="sum",
    )


def count_label_tokens(labels: torch.Tensor) -> int:
    """Number of positions that contribute to the loss (after the one-token shift)."""
    return int((labels[:, 1:] != IGNORE_INDEX).sum())


def no_decay(name: str) -> bool:
    """Trainer's rule: biases and normalisation weights get no weight decay."""
    name = name.lower()
    return "bias" in name or "norm" in name


def make_optimizer(model: nn.Module, cfg: LoopConfig) -> torch.optim.Optimizer:
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    groups = [
        {"params": [p for n, p in params if not no_decay(n)], "weight_decay": cfg.weight_decay},
        {"params": [p for n, p in params if no_decay(n)], "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=cfg.learning_rate, betas=(0.9, 0.999), eps=1e-8)


def train_raw(
    model: nn.Module, dataset, collator, cfg: LoopConfig, normalize: str = "tokens"
) -> dict[str, list[float]]:
    """Train with a hand-written loop. Returns per-step `loss`, `grad_norm` and `lr`.

    `normalize="tokens"` divides by the label tokens of the whole accumulated batch (what Trainer does).
    `normalize="micro_mean"` averages each micro-batch, then divides by `grad_accum`: the classic bug.
    """
    torch.manual_seed(cfg.seed)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, sampler=SequentialSampler(dataset), collate_fn=collator)
    optimizer = make_optimizer(model, cfg)
    scheduler = get_scheduler(
        "linear", optimizer, num_warmup_steps=cfg.warmup_steps, num_training_steps=cfg.max_steps
    )
    history: dict[str, list[float]] = {"loss": [], "grad_norm": [], "lr": []}
    model.train()
    batches = iter(loader)

    for step in range(cfg.max_steps):
        micro_batches = []
        for _ in range(cfg.grad_accum):
            try:
                micro_batches.append(next(batches))
            except StopIteration:  # start the next epoch, as Trainer does
                batches = iter(loader)
                micro_batches.append(next(batches))

        # Normalise by the label tokens of the whole accumulated batch, so every token has equal weight
        # no matter which micro-batch it landed in.
        n_tokens = sum(count_label_tokens(b["labels"]) for b in micro_batches)
        step_loss = 0.0
        for batch in micro_batches:
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
            if normalize == "tokens":
                loss = summed_token_loss(logits, batch["labels"]) / n_tokens
            else:
                loss = summed_token_loss(logits, batch["labels"]) / count_label_tokens(batch["labels"])
                loss = loss / cfg.grad_accum
            loss.backward()
            step_loss += loss.item()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        history["lr"].append(scheduler.get_last_lr()[0])
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        history["loss"].append(step_loss)
        history["grad_norm"].append(float(grad_norm))
    return history


def training_arguments(cfg: LoopConfig, output_dir: str, **overrides) -> TrainingArguments:
    """The `TrainingArguments` that match `LoopConfig`, grouped by what they control."""
    kwargs = dict(
        output_dir=output_dir,
        # batch
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        max_steps=cfg.max_steps,
        train_sampling_strategy="sequential",  # the raw loop does not shuffle; Trainer shuffles by default
        # optimisation
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_steps=cfg.warmup_steps,  # a float below 1 is a ratio; transformers 5 has no warmup_ratio
        lr_scheduler_type="linear",
        optim="adamw_torch",  # the default "adamw_torch_fused" differs in the last float bits
        max_grad_norm=cfg.max_grad_norm,
        # precision and device
        use_cpu=True,
        bf16=False,
        # logging, checkpointing, evaluation
        logging_steps=1,
        save_strategy="no",
        eval_strategy="no",
        report_to="none",
        disable_tqdm=True,
        seed=cfg.seed,
        dataloader_num_workers=0,
    )
    kwargs.update(overrides)
    return TrainingArguments(**kwargs)


def history_from_trainer(trainer: Trainer) -> dict[str, list[float]]:
    """Per-step logs as lists: `loss`, `grad_norm`, `lr`, plus any extra metric a subclass logged."""
    logs = [entry for entry in trainer.state.log_history if "loss" in entry and "train_runtime" not in entry]
    history: dict[str, list[float]] = {}
    for entry in logs:
        for key, value in entry.items():
            if key in ("epoch", "step"):
                continue
            history.setdefault("lr" if key == "learning_rate" else key, []).append(value)
    return history


def train_with_trainer(
    model: nn.Module, dataset, collator, cfg: LoopConfig, trainer_cls: type[Trainer] = Trainer, **trainer_kwargs
) -> dict[str, list[float]]:
    """Run the same configuration through `Trainer` (or a subclass). Returns the logged history."""
    with tempfile.TemporaryDirectory() as tmp:
        trainer = trainer_cls(
            model=model,
            args=training_arguments(cfg, tmp),
            train_dataset=dataset,
            data_collator=collator,
            **trainer_kwargs,
        )
        trainer.remove_callback(PrinterCallback)  # keep the console quiet; the history is returned
        trainer.train()
        return history_from_trainer(trainer)


def clone_model(model: nn.Module) -> nn.Module:
    """An independent copy with identical weights, so two loops can start from the same point."""
    return copy.deepcopy(model)
