"""A small SFT loop with an explicit per-token loss weight, so every masking and normalisation choice is visible."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import torch

from .lm import token_nll
from .templates import Encoded


@dataclass
class SFTConfig:
    lr: float = 2e-5
    epochs: int = 3
    batch_size: int = 16
    warmup_frac: float = 0.03
    weight_decay: float = 0.0
    norm: str = "token"          # "token": sum(w*nll)/sum(w) over the batch; "example": mean of per-example means
    seed: int = 0
    min_lr_frac: float = 0.1
    autocast: bool = True


def collate(batch: list[Encoded], pad_id: int, dev):
    L = max(len(e) for e in batch)
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    w = torch.zeros((len(batch), L))
    att = torch.zeros((len(batch), L), dtype=torch.long)
    for i, e in enumerate(batch):
        n = len(e)
        ids[i, :n] = torch.tensor(e.ids)
        w[i, :n] = torch.tensor(e.w)
        att[i, :n] = 1
    return ids.to(dev), w.to(dev), att.to(dev)


def weighted_loss(logits, ids, w, att, norm: str = "token") -> torch.Tensor:
    """Loss for next-token prediction. w[:, t] weights the term for predicting token t from tokens before it."""
    nll = token_nll(logits, ids)                       # (B, L-1), entry t predicts token t+1
    ww = w[:, 1:] * att[:, 1:]
    if norm == "token":
        return (nll * ww).sum() / ww.sum().clamp_min(1e-8)
    if norm == "example":
        per = (nll * ww).sum(1) / ww.sum(1).clamp_min(1e-8)
        return per[ww.sum(1) > 0].mean()
    raise ValueError(norm)


def lr_at(step: int, total: int, cfg: SFTConfig) -> float:
    warm = max(1, int(cfg.warmup_frac * total))
    if step < warm:
        return cfg.lr * (step + 1) / warm
    p = (step - warm) / max(1, total - warm)
    return cfg.lr * (cfg.min_lr_frac + (1 - cfg.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * p)))


def train(model, data: list[Encoded], pad_id: int, cfg: SFTConfig, on_epoch=None, log_every: int = 0) -> list[dict]:
    """Train in place. `on_epoch(epoch, model)` runs after each epoch and may return a dict that is logged."""
    dev = next(model.parameters()).device
    rng = random.Random(cfg.seed)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95))
    steps_per_epoch = math.ceil(len(data) / cfg.batch_size)
    total = steps_per_epoch * cfg.epochs
    step, log = 0, []
    model.train()
    for ep in range(cfg.epochs):
        order = list(range(len(data)))
        rng.shuffle(order)
        run, cnt = 0.0, 0
        for b in range(0, len(order), cfg.batch_size):
            batch = [data[i] for i in order[b:b + cfg.batch_size]]
            ids, w, att = collate(batch, pad_id, dev)
            for g in opt.param_groups:
                g["lr"] = lr_at(step, total, cfg)
            with torch.autocast(dev if dev == "cuda" else "cpu", dtype=torch.bfloat16, enabled=cfg.autocast and dev == "cuda"):
                logits = model(input_ids=ids, attention_mask=att).logits
            loss = weighted_loss(logits, ids, w, att, cfg.norm)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            run += loss.item()
            cnt += 1
            step += 1
        rec = {"epoch": ep + 1, "train_loss": run / cnt, "steps": step}
        if on_epoch is not None:
            model.eval()
            extra = on_epoch(ep + 1, model)
            model.train()
            if extra:
                rec.update(extra)
        log.append(rec)
    model.eval()
    return log


@torch.no_grad()
def mean_nll(model, data: list[Encoded], pad_id: int, batch_size: int = 32) -> dict:
    """Weighted per-token NLL over a dataset: the token-level mean (all scored tokens count equally) and the
    example-level mean (every example counts equally)."""
    dev = next(model.parameters()).device
    tot, wt, per = 0.0, 0.0, []
    model.eval()
    for b in range(0, len(data), batch_size):
        batch = data[b:b + batch_size]
        ids, w, att = collate(batch, pad_id, dev)
        with torch.autocast(dev if dev == "cuda" else "cpu", dtype=torch.bfloat16, enabled=dev == "cuda"):
            logits = model(input_ids=ids, attention_mask=att).logits
        nll = token_nll(logits, ids)
        ww = w[:, 1:] * att[:, 1:]
        tot += float((nll * ww).sum())
        wt += float(ww.sum())
        s = ww.sum(1)
        per += ((nll * ww).sum(1) / s.clamp_min(1e-8))[s > 0].tolist()
    return {"token_nll": tot / max(wt, 1e-8), "example_nll": sum(per) / max(len(per), 1), "tokens": wt}
