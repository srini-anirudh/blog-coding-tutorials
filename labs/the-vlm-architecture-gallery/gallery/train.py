"""Training and evaluation loops for the gallery experiments. Small on purpose: CPU, minutes."""

from __future__ import annotations

import copy
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .data import VOCAB, GlyphPages
from .fusion import TinyLM
from .model import TinyVLM

TEXT_TASKS = ("fine", "coarse", "add")


def _step(model_fn, params, opt, batch_fn, steps, g, log_every=0, label=""):
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 50) * 0.5 * (1 + torch.cos(torch.tensor(s / steps * 3.14159)).item()))
    t0 = time.time()
    for s in range(steps):
        inputs, ans = batch_fn(s, g)
        loss = F.cross_entropy(model_fn(*inputs), ans)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if log_every and (s + 1) % log_every == 0:
            print(f"  {label} step {s + 1:5d}  loss {loss.item():.3f}  {time.time() - t0:.0f}s", flush=True)
    return time.time() - t0


def pretrain_lm(seed: int = 0, steps: int = 3000, batch: int = 128, lr: float = 2e-3, cache: Path | None = None,
                log_every: int = 0, **lm_kw) -> TinyLM:
    """Text-only pretraining on the three text tasks. Cached to `cache` if given (not committed)."""
    if cache is not None and cache.exists():
        lm = TinyLM(VOCAB, **lm_kw)
        lm.load_state_dict(torch.load(cache, weights_only=True))
        return lm
    torch.manual_seed(seed)
    lm = TinyLM(VOCAB, **lm_kw)
    data = GlyphPages(seed=0)
    g = torch.Generator().manual_seed(1000 + seed)
    opt = torch.optim.AdamW(lm.parameters(), lr=lr, weight_decay=0.01)

    def batch_fn(s, g):
        ids, ans = data.text_batch(batch, TEXT_TASKS[s % 3], g)
        return (ids,), ans

    _step(lambda ids: lm(ids)[:, -1], list(lm.parameters()), opt, batch_fn, steps, g, log_every, "lm")
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(lm.state_dict(), cache)
    return lm


def set_trainable(vlm: TinyVLM, schedule: str) -> list[torch.nn.Parameter]:
    """Freezing schedules from the post's fourth question.

    "connector"  vision encoder and LM frozen; connector (+ any new xattn / expert weights) trains.
    "all"        vision encoder frozen; connector and the whole LM train.
    The encoder is a fixed random stand-in, so it never trains.
    """
    vlm.requires_grad_(False)
    vlm.connector.requires_grad_(True)
    if schedule == "all":
        vlm.lm.requires_grad_(True)
    elif schedule != "connector":
        raise ValueError(schedule)
    for m in vlm.new_lm_modules:
        m.requires_grad_(True)
    return [p for p in vlm.parameters() if p.requires_grad]


def train_vlm(vlm: TinyVLM, tasks: tuple[str, ...], schedule: str, steps: int = 1500, batch: int = 128,
              lr: float = 1e-3, seed: int = 0, log_every: int = 0) -> dict:
    params = set_trainable(vlm, schedule)
    data = GlyphPages(seed=0)
    g = torch.Generator().manual_seed(2000 + seed)
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)

    def batch_fn(s, g):
        img, q, ans = data.visual_batch(batch, tasks[s % len(tasks)], g)
        return (img, q), ans

    vlm.train()
    secs = _step(vlm, params, opt, batch_fn, steps, g, log_every, "vlm")
    return {"train_seconds": round(secs, 1), "trainable_params": sum(p.numel() for p in params)}


@torch.no_grad()
def evaluate(vlm_or_lm, task: str, visual: bool, n: int = 2048, seed: int = 99, batch: int = 512) -> float:
    """Accuracy on n fresh examples. `visual=False` runs the text-only version through the LM."""
    data = GlyphPages(seed=0)
    g = torch.Generator().manual_seed(seed)
    vlm_or_lm.eval()
    correct = 0
    for _ in range(n // batch):
        if visual:
            img, q, ans = data.visual_batch(batch, task, g)
            logits = vlm_or_lm(img, q)
        else:
            ids, ans = data.text_batch(batch, task, g)
            lm = vlm_or_lm.lm if isinstance(vlm_or_lm, TinyVLM) else vlm_or_lm
            logits = lm(ids)[:, -1]
        correct += (logits.argmax(-1) == ans).sum().item()
    return correct / (n // batch * batch)


def text_scores(model) -> dict:
    return {t: evaluate(model, t, visual=False) for t in TEXT_TASKS}


def fresh_copy(lm: TinyLM) -> TinyLM:
    return copy.deepcopy(lm)
