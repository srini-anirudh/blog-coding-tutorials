"""Supervised fine-tuning of the tiny VLM, cached in runs/, plus shared evaluation helpers."""

from __future__ import annotations

import os
import random
import time
from pathlib import Path

import torch

from .metrics import score_response, summarise
from .model import TinyVLM, pad, sample, sft_loss
from .world import decode, encode, null_image, render, sample_scene, style_for

LAB = Path(__file__).resolve().parents[1]
RUNS = LAB / "runs"
NOISE = 0.8  # evidence noise used everywhere unless a script says otherwise


def set_threads() -> None:
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", min(8, os.cpu_count() or 1))))


def sft_data(n: int, seed: int, noise: float = NOISE, null_frac: float = 0.1, style_p: float = 0.3):
    """n (image, description) pairs. Claim order is random; a fraction see the null image."""
    rng, gen = random.Random(seed), torch.Generator().manual_seed(seed)
    imgs, seqs = [], []
    for _ in range(n):
        s = sample_scene(rng)
        claims = s.claims()
        rng.shuffle(claims)
        seqs.append(encode(claims, style_for(len(claims), rng, style_p)))
        imgs.append(null_image() if rng.random() < null_frac else render(s, gen, noise))
    return torch.stack(imgs), pad(seqs)


def train_sft(n: int = 40000, epochs: int = 8, bs: int = 128, lr: float = 2e-3, seed: int = 0,
              log=print) -> TinyVLM:
    torch.manual_seed(seed)
    model = TinyVLM()
    imgs, ids = sft_data(n, seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * (n // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.1)
    t0, step = time.time(), 0
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n - bs + 1, bs):
            j = perm[i:i + bs]
            loss = sft_loss(model, imgs[j], ids[j])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            step += 1
        log(f"[sft] epoch {ep} loss={loss.item():.4f} t={time.time() - t0:.0f}s")
    return model.eval()


def load_sft(log=print) -> TinyVLM:
    """Train once, then reuse runs/sft.pt. Every script that needs the base policy calls this."""
    path = RUNS / "sft.pt"
    if path.exists():
        m = TinyVLM()
        m.load_state_dict(torch.load(path, weights_only=True))
        return m.eval()
    m = train_sft(log=log)
    RUNS.mkdir(exist_ok=True)
    torch.save(m.state_dict(), path)
    return m


def eval_scenes(n: int, seed: int, noise: float = NOISE):
    rng, gen = random.Random(seed), torch.Generator().manual_seed(seed)
    scenes = [sample_scene(rng) for _ in range(n)]
    return scenes, torch.stack([render(s, gen, noise) for s in scenes])


@torch.no_grad()
def describe_eval(model: TinyVLM, scenes, imgs, bs: int = 250) -> dict[str, float]:
    """Greedy descriptions scored against the truth: CHAIR, recall, attribute and count errors."""
    rows, n_style = [], 0
    for i in range(0, len(scenes), bs):
        for s, ids in zip(scenes[i:i + bs], sample(model, imgs[i:i + bs], greedy=True)):
            claims, st = decode(ids)
            rows.append(score_response(s, claims))
            n_style += st
    out = summarise(rows)
    out["style_per_response"] = n_style / len(scenes)
    return out
