"""One short TinyLLM training run with any optimizer, plus the update-to-weight RMS diagnostic."""

from __future__ import annotations

import math
import time

import torch

from .common import shakespeare_batches, tiny_model
from .optimizers import rms


def set_lr(opt, lr_scale: float, base: dict) -> None:
    for o in getattr(opt, "opts", [opt]):
        o.lr = base[id(o)] * lr_scale


@torch.no_grad()
def evaluate(model, batches=10, batch=16, seq=128) -> float:
    data = shakespeare_batches(batch, seq, seed=123, split="val")
    return sum(model(x, labels=x).loss.item() for x, _ in zip(data, range(batches))) / batches


def train(make_opt, steps=300, batch=16, seq=128, warmup=30, seed=0, clip=1.0, log_every=50):
    """Linear warmup, then cosine decay to 10% of the peak lr. Returns a dict of results.

    make_opt(model) builds the optimizer. The update-to-weight ratio RMS(dW)/RMS(W) is measured per
    2-D tensor on the last step.
    """
    model = tiny_model(seed=seed, seq=seq)
    opt = make_opt(model)
    base = {id(o): o.lr for o in getattr(opt, "opts", [opt])}
    data = shakespeare_batches(batch, seq, seed=seed)
    curve, t0 = [], time.perf_counter()
    for step in range(steps):
        frac = step / max(1, steps - 1)
        scale = (step + 1) / warmup if step < warmup else 0.1 + 0.45 * (1 + math.cos(math.pi * frac))
        set_lr(opt, scale, base)
        x = next(data)
        loss = model(x, labels=x).loss
        opt.zero_grad()
        loss.backward()
        if clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        last = step == steps - 1
        if last:
            before = {n: p.detach().clone() for n, p in model.named_parameters() if p.ndim == 2}
        opt.step()
        if not math.isfinite(loss.item()):
            return {"diverged": True, "step": step, "curve": curve}
        if step % log_every == 0 or last:
            curve.append((step, round(loss.item(), 4)))
    ratios = {n: (rms(p.detach() - before[n]) / rms(p.detach())).item()
              for n, p in model.named_parameters() if p.ndim == 2}
    return {
        "diverged": False,
        "train_loss": curve[-1][1],
        "val_loss": evaluate(model, seq=seq),
        "curve": curve,
        "seconds": time.perf_counter() - t0,
        "state_bytes_per_param": opt.state_bytes() / sum(p.numel() for p in model.parameters()),
        "update_ratio": ratios,
        "model": model,
        "opt": opt,
    }
