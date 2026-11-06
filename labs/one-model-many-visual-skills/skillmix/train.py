"""Batches from a mixture, the masked SFT loss with two normalizations, freezing strategies, generation."""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from .chat import IM_END, PAD, VOCAB, serialize
from .model import N_VISUAL, TinyVLM, add_lora
from .scenes import IMAGE_SIZE, render
from .tasks import make_example

MAX_LEN = 96


def collate(examples: list) -> dict[str, torch.Tensor]:
    """examples: [(turns, scene_or_None, source)] -> right-padded ids, targets, mask, images."""
    seqs = [serialize(turns, N_VISUAL) for turns, _, _ in examples]
    t = max(len(s.ids) for s in seqs)
    assert t <= MAX_LEN, t
    ids = torch.full((len(seqs), t), PAD)
    mask = torch.zeros(len(seqs), t)
    images = torch.zeros(len(seqs), 3, IMAGE_SIZE, IMAGE_SIZE)
    for i, (s, (_, scene, _)) in enumerate(zip(seqs, examples)):
        ids[i, :len(s.ids)] = torch.tensor(s.ids)
        mask[i, :len(s.mask)] = torch.tensor(s.mask, dtype=torch.float)
        if scene is not None:
            images[i] = render(scene)
    return {"ids": ids, "mask": mask, "images": images, "sources": [src for _, _, src in examples]}


def sft_loss(logits: torch.Tensor, ids: torch.Tensor, mask: torch.Tensor, norm: str = "token") -> torch.Tensor:
    """Next-token loss on masked positions.

    token:   sum over every target token in the batch / number of target tokens.
             A source's share of the signal = its sampling weight x its answer length.
    example: mean over each example's target tokens, then mean over examples.
             Every example counts the same, however long its answer is.
    """
    nll = F.cross_entropy(logits[:, :-1].transpose(1, 2), ids[:, 1:], reduction="none")
    m = mask[:, 1:]
    if norm == "token":
        return (nll * m).sum() / m.sum().clamp(min=1)
    if norm == "example":
        per = (nll * m).sum(1) / m.sum(1).clamp(min=1)
        return per[m.sum(1) > 0].mean()
    raise ValueError(norm)


def token_weights(mask: torch.Tensor, norm: str) -> torch.Tensor:
    """(B,) the total weight each example's tokens receive inside sft_loss. Sums to 1."""
    n = mask[:, 1:].sum(1)
    w = n / n.sum() if norm == "token" else torch.ones_like(n) / len(n)
    return w


@dataclass
class RunConfig:
    mixture: dict[str, float]
    steps: int = 1000
    batch: int = 32
    lr: float = 2e-3
    norm: str = "token"
    fmt: bool = False            # format prompt on vqa questions
    teacher_bias: float = 0.0    # chance the caption teacher adds an absent fork
    verify: bool = False         # drop caption sentences that name absent objects
    strategy: str = "full"       # full | low_llm_lr | lora | frozen_llm
    llm_lr_scale: float = 0.1
    seed: int = 0
    log_every: int = 0
    extra: dict = field(default_factory=dict)


def sample_batch(cfg: RunConfig, rng: random.Random) -> list:
    names, weights = zip(*cfg.mixture.items())
    out = []
    for src in rng.choices(names, weights=weights, k=cfg.batch):
        turns, scene = make_example(src, rng, fmt=cfg.fmt, teacher_bias=cfg.teacher_bias, verify=cfg.verify)
        out.append((turns, scene, src))
    return out


def param_groups(model: TinyVLM, cfg: RunConfig) -> list[dict]:
    dec = {id(p) for m in model.decoder_modules() for p in m.parameters()}
    if cfg.strategy == "lora":
        add_lora(model)
    if cfg.strategy == "frozen_llm":
        for m in model.decoder_modules():
            for p in m.parameters():
                p.requires_grad_(False)
    trainable = [p for p in model.parameters() if p.requires_grad]
    if cfg.strategy == "low_llm_lr":
        return [{"params": [p for p in trainable if id(p) not in dec], "lr": cfg.lr},
                {"params": [p for p in trainable if id(p) in dec], "lr": cfg.lr * cfg.llm_lr_scale}]
    return [{"params": trainable, "lr": cfg.lr}]


def train(model: TinyVLM, cfg: RunConfig) -> dict:
    """Train in place. Returns the loss curve and wall time."""
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)
    opt = torch.optim.AdamW(param_groups(model, cfg), weight_decay=0.0)
    base_lrs = [g["lr"] for g in opt.param_groups]
    model.train()
    curve, t0 = [], time.time()
    for step in range(cfg.steps):
        frac = step / max(1, cfg.steps)
        for g, lr in zip(opt.param_groups, base_lrs):  # short warmup, then cosine decay
            g["lr"] = lr * min(1.0, (step + 1) / 50) * 0.5 * (1 + math.cos(math.pi * frac))
        b = collate(sample_batch(cfg, rng))
        loss = sft_loss(model(b["ids"], b["images"]), b["ids"], b["mask"], cfg.norm)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if cfg.log_every and (step % cfg.log_every == 0 or step == cfg.steps - 1):
            curve.append((step, round(loss.item(), 4)))
    model.eval()
    return {"curve": curve, "seconds": round(time.time() - t0, 1)}


@torch.no_grad()
def generate(model: TinyVLM, prompts: list[tuple[list, object]], max_new: int = 60) -> list[str]:
    """Greedy decoding. prompts: [(turns_without_answer, scene_or_None)]. Groups equal-length prompts."""
    model.eval()
    seqs = [serialize(turns, N_VISUAL, add_generation_prompt=True).ids for turns, _ in prompts]
    out: list[str] = [""] * len(prompts)
    by_len: dict[int, list[int]] = {}
    for i, s in enumerate(seqs):
        by_len.setdefault(len(s), []).append(i)
    for _, idx in by_len.items():
        for start in range(0, len(idx), 256):
            chunk = idx[start:start + 256]
            ids = torch.tensor([seqs[i] for i in chunk])
            images = torch.stack([render(prompts[i][1]) if prompts[i][1] is not None
                                  else torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE) for i in chunk])
            done = torch.zeros(len(chunk), dtype=torch.bool)
            gen = []
            for _ in range(min(max_new, MAX_LEN - ids.shape[1])):
                nxt = model(ids, images)[:, -1].argmax(-1)
                nxt = torch.where(done, torch.full_like(nxt, IM_END), nxt)
                gen.append(nxt)
                done |= nxt == IM_END
                ids = torch.cat([ids, nxt[:, None]], 1)
                if done.all():
                    break
            gen_t = torch.stack(gen, 1).tolist()
            for row, i in zip(gen_t, chunk):
                row = row[:row.index(IM_END)] if IM_END in row else row
                out[i] = VOCAB.decode(row)
    return out


def new_model(seed: int = 0) -> TinyVLM:
    torch.manual_seed(seed)
    return TinyVLM(len(VOCAB), max_len=MAX_LEN)
