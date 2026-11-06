"""The pretraining loop. Every piece it calls is a small module in this package."""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import torch

from pretrain.activation_ckpt import apply_activation_checkpointing
from pretrain.checkpoint import latest_valid_checkpoint, load_checkpoint, save_checkpoint
from pretrain.config import TrainConfig
from pretrain.distributed import DistInfo, mean_across_ranks, unwrap, wrap_ddp
from pretrain.guards import NonFiniteGuard, SpikeDetector, check_batch, step_with_oom_retry
from pretrain.loader import PackedWindows, ResumableLoader
from pretrain.metrics import MetricsLogger, param_norm
from pretrain.mfu import flops_per_token
from pretrain.model import build_model, count_params
from pretrain.optim import build_optimizer
from pretrain.schedule import build_scheduler
from pretrain.shards import TokenShards
from pretrain.step import autocast, lm_loss_sum, train_step
from pretrain.utils import memory_mb, pick_device, resolve, seed_everything, synchronize


@torch.no_grad()
def evaluate(model, val: PackedWindows, n_batches: int, batch_size: int, device, precision: str) -> float:
    """Mean loss on a fixed, evenly spaced set of validation windows (same set every time)."""
    model.eval()
    idx = np.linspace(0, len(val) - 1, n_batches * batch_size).astype(int)
    loss, tokens = 0.0, 0
    for b in range(n_batches):
        xs, ys = zip(*(val[int(i)] for i in idx[b * batch_size : (b + 1) * batch_size]))
        x, y = torch.stack(xs).to(device), torch.stack(ys).to(device)
        with autocast(device, precision):
            logits = model(x).logits
        loss += lm_loss_sum(logits, y).item()
        tokens += y.numel()
    model.train()
    return loss / tokens


def empty_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def train(cfg: TrainConfig, dist_info: DistInfo | None = None, resume: bool = True,
          stop_after: int | None = None, peak_flops: float | None = None) -> list[dict]:
    """Train until cfg.max_steps (or `stop_after` steps in this call). Returns logged records."""
    info = dist_info or DistInfo()
    device = pick_device(cfg.device) if not info.enabled else torch.device(
        f"cuda:{info.local_rank}" if torch.cuda.is_available() else "cpu")
    out_dir = resolve(cfg.out_dir)
    seed_everything(cfg.seed)  # identical init on every rank (DDP also broadcasts rank 0)

    # ---- data
    shards = TokenShards(resolve(cfg.data_dir), "train")
    assert shards.vocab_size <= cfg.model.vocab_size, "tokenizer vocab larger than model vocab"
    train_ds = PackedWindows(shards, cfg.seq_len)
    val_ds = PackedWindows(TokenShards(resolve(cfg.data_dir), "val"), cfg.seq_len)
    loader = ResumableLoader(train_ds, cfg.micro_batch_size, cfg.seed, info.rank, info.world_size,
                             num_workers=cfg.num_workers, pin_memory=device.type == "cuda")

    # ---- model, optimiser, schedule
    model = build_model(cfg.model, cfg.model_type)
    if cfg.activation_checkpointing:
        apply_activation_checkpointing(model)
    model.to(device)
    opt = build_optimizer(model, cfg, device)
    sched = build_scheduler(opt, cfg.warmup_steps, cfg.max_steps, cfg.lr, cfg.min_lr)
    spikes, nonfinite = SpikeDetector(), NonFiniteGuard()

    step = 0
    ckpt = latest_valid_checkpoint(out_dir) if resume else None
    if ckpt is not None:
        state = load_checkpoint(ckpt, model, opt, sched, loader, device)
        assert state["config"] == cfg.to_dict(), "config changed since the checkpoint was written"
        step = state["step"]
        spikes.load_state_dict(state["extra"]["spikes"])
        print(f"[resume] {ckpt.name}: step {step}, data epoch {loader.epoch} batch {loader.consumed}")
    ddp_model = wrap_ddp(model, info, device)

    log = MetricsLogger(out_dir, info.rank)
    fpt = flops_per_token(model, cfg.model, cfg.seq_len)
    if info.is_main and step == 0:
        log.log({"event": "start", "params": count_params(model), "flops_per_token": fpt,
                 "tokens_per_step": cfg.tokens_per_step * info.world_size, "device": str(device),
                 "world_size": info.world_size})

    def save() -> None:
        if info.is_main:
            save_checkpoint(out_dir, step, unwrap(model), opt, sched, loader, cfg.to_dict(),
                            extra={"spikes": spikes.state_dict()}, keep=cfg.keep_checkpoints)

    t_start, records, steps_this_call = time.time(), [], 0
    t_last, tokens_since, steps_since = time.time(), 0, 0
    while step < cfg.max_steps:
        micro = [next(loader) for _ in range(cfg.grad_accum_steps)]
        for x, y in micro:
            check_batch(x, y, cfg.model.vocab_size)

        lr = sched.get_last_lr()[0]  # the LR this step uses
        res, n_micro = step_with_oom_retry(
            lambda mb: train_step(ddp_model, opt, sched, mb, device, cfg.precision, cfg.grad_clip),
            micro, opt, lambda: empty_cache(device))
        nonfinite.update(res.skipped)  # raises TrainingDiverged after `patience` in a row
        spike = spikes.update(res.loss)
        step += 1
        steps_this_call += 1
        tokens_since += res.tokens
        steps_since += 1

        if step % cfg.log_every == 0 or step == 1 or spike or res.skipped:
            synchronize(device)
            dt = time.time() - t_last
            tok_s = mean_across_ranks(tokens_since / dt, device) * info.world_size
            rec = {"step": step, "loss": mean_across_ranks(res.loss, device),
                   "lr": lr, "grad_norm": res.grad_norm,
                   "param_norm": param_norm(model), "tok_s": tok_s,
                   "step_ms": 1000 * dt / steps_since,
                   "mem_mb": memory_mb(device), "epoch": loader.epoch}
            if peak_flops:
                rec["mfu"] = tok_s / info.world_size * fpt / peak_flops
            if spike:
                rec["spike"] = True
            if res.skipped:
                rec["skipped"] = True
            if n_micro != cfg.grad_accum_steps:
                rec["micro_batches"] = n_micro
            log.log(rec)
            records.append(rec)
            t_last, tokens_since, steps_since = time.time(), 0, 0

        if step % cfg.eval_every == 0 or step == cfg.max_steps:
            val = evaluate(model, val_ds, cfg.eval_batches, cfg.micro_batch_size, device, cfg.precision)
            rec = {"step": step, "val_loss": mean_across_ranks(val, device),
                   "val_ppl": math.exp(val), "tokens_seen": step * cfg.tokens_per_step * info.world_size}
            log.log(rec)
            records.append(rec)
            t_last = time.time()  # eval time is not training throughput

        out_of_time = cfg.max_minutes and (time.time() - t_start) / 60 > cfg.max_minutes
        if step % cfg.ckpt_every == 0 or step == cfg.max_steps or out_of_time:
            save()
            t_last = time.time()  # checkpoint time is not training throughput either
        if out_of_time or (stop_after is not None and steps_this_call >= stop_after):
            break
    return records
