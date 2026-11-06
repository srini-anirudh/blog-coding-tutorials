"""Train a TwoTowerVLM on the glyph-board task and track loss against training FLOPs."""

from __future__ import annotations

import time

import torch

from . import data
from .model import TwoTowerConfig, TwoTowerVLM, matmul_params


def train_flops_per_sample(model: TwoTowerVLM) -> float:
    """6 x matmul parameters x tokens, per tower (all towers trained; attention scores left out)."""
    cfg = model.cfg
    return 6.0 * (matmul_params(model.vision) * cfg.patches
                  + matmul_params(model.connector) * cfg.visual_tokens
                  + matmul_params(model.llm) * (cfg.visual_tokens + data.TEXT_LEN))


@torch.no_grad()
def evaluate(model: TwoTowerVLM, batches: list[dict], device: torch.device) -> dict[str, float]:
    model.eval()
    out = {}
    for task, batch in batches:
        px, txt, tg = (batch[k].to(device) for k in ("pixels", "text", "targets"))
        logits = model(px, txt)[:, 1]
        out[f"{task}_loss"] = float(torch.nn.functional.cross_entropy(logits, tg[:, 1]))
        out[f"{task}_acc"] = float((logits.argmax(-1) == tg[:, 1]).float().mean())
    model.train()
    return out


def train(cfg: TwoTowerConfig, steps: int = 2000, batch: int = 128, lr: float = 2e-3, eval_every: int = 250,
          eval_size: int = 1024, seed: int = 0, device: str = "cpu") -> dict:
    torch.manual_seed(seed)
    dev = torch.device(device)
    model = TwoTowerVLM(cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 100) * max(0.05, 1 - s / steps))
    gen = torch.Generator().manual_seed(seed + 1)
    eval_gen = torch.Generator().manual_seed(10_000 + seed)
    eval_batches = [(t, data.make_batch(eval_size, eval_gen, t)) for t in ("color", "glyph")]
    per_sample = train_flops_per_sample(model)
    history, t0 = [], time.time()
    for step in range(1, steps + 1):
        b = data.make_batch(batch, gen)
        loss = model.loss(b["pixels"].to(dev), b["text"].to(dev), b["targets"].to(dev))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % eval_every == 0 or step == steps:
            row = {"step": step, "train_flops": per_sample * batch * step, "train_loss": float(loss.detach())}
            row.update(evaluate(model, eval_batches, dev))
            history.append(row)
    return {
        "config": {k: getattr(cfg, k) for k in cfg.__dataclass_fields__},
        "visual_tokens": cfg.visual_tokens,
        "params": {"vision": matmul_params(model.vision), "connector": matmul_params(model.connector),
                   "llm": matmul_params(model.llm)},
        "train_flops_per_sample": per_sample,
        "seconds": time.time() - t0,
        "history": history,
        "final": history[-1],
    }
