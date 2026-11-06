"""Training loops and evaluation for the three grounding interfaces on the toy referring task."""

from __future__ import annotations

import copy
import math
import time

import numpy as np
import torch

from .models import DecodeGrounder, SelectGrounder, TypedGrounder, bce_dice
from .toy import IRREGULAR, Batch, batch_box_iou, box_masks, proposals, subset

# Data stays on the CPU. Each training batch and each evaluation chunk is moved to DEVICE.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _dev(b: Batch) -> Batch:
    return Batch(b.image.to(DEVICE), b.query.to(DEVICE), b.mask.to(DEVICE), b.box.to(DEVICE), b.kind,
                 b.other_boxes, b.touching)


def _cosine(opt, step, total, lr, warmup=50):
    f = step / warmup if step < warmup else 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))
    for g in opt.param_groups:
        g["lr"] = lr * f


def _batches(n, batch, steps, seed):
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        yield torch.randint(0, n, (batch,), generator=g)


def _log(log, msg):
    if log is not None:
        log(msg)


# ---- Type ------------------------------------------------------------------------------------

def train_typed(fmt: str, data: Batch, steps: int, batch: int = 64, lr: float = 2e-3, seed: int = 0, log=None):
    torch.manual_seed(seed)
    model = TypedGrounder(fmt).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr, weight_decay=0.01)
    t0 = time.time()
    for step, idx in enumerate(_batches(len(data.query), batch, steps, seed)):
        _cosine(opt, step, steps, lr)
        b = _dev(subset(data, idx))
        loss = model.nll(b.image, b.query, b.box).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 250 == 0 or step == steps - 1:
            _log(log, f"  [{fmt} CE] step {step:5d}  nll/box {loss.item():.3f}  {time.time() - t0:.0f}s")
    return model


def rl_finetune(model: TypedGrounder, data: Batch, steps: int, images: int = 16, group: int = 8,
                lr: float = 1e-4, seed: int = 0, log=None):
    """GRPO-style policy gradient: sample `group` boxes per image, reward = IoU with the true box,
    advantage = reward minus the group mean, divided by the group standard deviation. No KL term."""
    torch.manual_seed(seed)
    model = copy.deepcopy(model)
    opt = torch.optim.AdamW(model.parameters(), lr, weight_decay=0.0)
    t0 = time.time()
    for step, idx in enumerate(_batches(len(data.query), images, steps, seed + 1)):
        b = _dev(subset(data, idx.repeat_interleave(group)))
        box, logp = model.predict(b.image, b.query, sample=True)
        r = batch_box_iou(box, b.box).view(images, group)
        adv = ((r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-4)).view(-1)
        loss = -(adv.detach() * logp).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 50 == 0 or step == steps - 1:
            _log(log, f"  [{model.fmt} RL] step {step:4d}  mean sampled IoU {r.mean().item():.3f}  {time.time() - t0:.0f}s")
    return model


@torch.no_grad()
def predict_boxes(model: TypedGrounder, data: Batch, chunk: int = 250):
    model.eval()
    out = [model.predict(data.image[i:i + chunk].to(DEVICE), data.query[i:i + chunk].to(DEVICE))[0].cpu()
           for i in range(0, len(data.query), chunk)]
    model.train()
    return torch.cat(out)


@torch.no_grad()
def coordinate_nll(model: TypedGrounder, data: Batch, chunk: int = 250) -> float:
    """Mean negative log-likelihood of one true coordinate, in nats (a digit coordinate sums its three digits)."""
    model.eval()
    tot = sum(model.nll(*(t[i:i + chunk].to(DEVICE) for t in (data.image, data.query, data.box))).sum().item()
              for i in range(0, len(data.query), chunk))
    model.train()
    return tot / (4 * len(data.query))


def box_metrics(pred: torch.Tensor, data: Batch, bins: int = 1000) -> dict:
    iou = batch_box_iou(pred, data.box)
    wrong = torch.tensor([any(_iou1(pred[i], o) >= 0.5 for o in data.other_boxes[i]) for i in range(len(pred))])
    err = ((pred - data.box) / 64 * bins).abs()                     # in bins of the 0-1000 grid
    mask_iou = _mask_iou(box_masks(pred) > 0, data.mask > 0)
    irr = torch.tensor([k in IRREGULAR for k in data.kind])
    return {"box_iou": iou.mean().item(), "acc50": (iou >= 0.5).float().mean().item(),
            "acc75": (iou >= 0.75).float().mean().item(), "wrong_object": wrong.float().mean().item(),
            "far_miss_coords": (err > 100).float().mean().item(), "median_err_bins": err.median().item(),
            "mask_iou_of_box_compact": mask_iou[~irr].mean().item(),
            "mask_iou_of_box_irregular": mask_iou[irr].mean().item()}


def _iou1(a, b):
    return batch_box_iou(a[None], torch.tensor(b, dtype=torch.float32)[None]).item()


def _mask_iou(p: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    inter = (p & g).flatten(1).sum(1).float()
    union = (p | g).flatten(1).sum(1).float()
    return torch.where(union > 0, inter / union.clamp(min=1), torch.ones_like(union))


def mask_metrics(pred: torch.Tensor, data: Batch) -> dict:
    """Mask IoU, gIoU and cIoU, Acc@0.5, split by compact vs irregular shapes and by separated vs touching scenes."""
    p, g = pred > 0, data.mask > 0
    iou = _mask_iou(p, g)
    irr = torch.tensor([k in IRREGULAR for k in data.kind])
    inter = (p & g).flatten(1).sum(1).float()
    union = (p | g).flatten(1).sum(1).float()
    out = {"gIoU": iou.mean().item(), "cIoU": (inter.sum() / union.sum()).item(),
           "acc50": (iou >= 0.5).float().mean().item()}
    for name, sel in [("compact", ~irr), ("irregular", irr), ("separated", ~data.touching), ("touching", data.touching)]:
        out[f"gIoU_{name}"] = iou[sel].mean().item()
    return out


# ---- Select ----------------------------------------------------------------------------------

def proposal_targets(props, data: Batch):
    """Index of the proposal that best matches the true mask, and that best IoU (the proposer's ceiling)."""
    tgt, ceil = [], []
    for i, regions in enumerate(props):
        if not regions:
            tgt.append(0)
            ceil.append(0.0)
            continue
        ious = _mask_iou(torch.stack(regions), (data.mask[i] > 0).expand(len(regions), -1, -1))
        tgt.append(int(ious.argmax()))
        ceil.append(float(ious.max()))
    return torch.tensor(tgt), torch.tensor(ceil)


def train_select(data: Batch, steps: int, batch: int = 64, lr: float = 2e-3, seed: int = 0, log=None):
    torch.manual_seed(seed)
    props = proposals(data.image)
    tgt, _ = proposal_targets(props, data)
    model = SelectGrounder().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr, weight_decay=0.01)
    t0 = time.time()
    for step, idx in enumerate(_batches(len(data.query), batch, steps, seed)):
        _cosine(opt, step, steps, lr)
        b = _dev(subset(data, idx))
        s = model.scores(b.image, b.query, [props[i] for i in idx.tolist()])
        loss = torch.nn.functional.cross_entropy(s, tgt[idx].to(DEVICE))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 250 == 0 or step == steps - 1:
            _log(log, f"  [select] step {step:5d}  CE over proposals {loss.item():.3f}  {time.time() - t0:.0f}s")
    return model


@torch.no_grad()
def predict_select(model: SelectGrounder, data: Batch, chunk: int = 250):
    props = proposals(data.image)
    masks = torch.zeros_like(data.mask)
    for i in range(0, len(data.query), chunk):
        s = model.scores(data.image[i:i + chunk].to(DEVICE), data.query[i:i + chunk].to(DEVICE), props[i:i + chunk]).cpu()
        for j, k in enumerate(s.argmax(1).tolist()):
            if props[i + j]:
                masks[i + j] = props[i + j][k].float()
    _, ceil = proposal_targets(props, data)
    return masks, ceil


# ---- Decode ----------------------------------------------------------------------------------

def train_decode(data: Batch, steps: int, batch: int = 64, lr: float = 2e-3, seed: int = 0, log=None):
    torch.manual_seed(seed)
    model = DecodeGrounder().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr, weight_decay=0.01)
    t0 = time.time()
    for step, idx in enumerate(_batches(len(data.query), batch, steps, seed)):
        _cosine(opt, step, steps, lr)
        b = _dev(subset(data, idx))
        loss = bce_dice(model(b.image, b.query), b.mask)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 250 == 0 or step == steps - 1:
            _log(log, f"  [decode] step {step:5d}  bce+dice {loss.item():.3f}  {time.time() - t0:.0f}s")
    return model


@torch.no_grad()
def predict_decode(model: DecodeGrounder, data: Batch, chunk: int = 250):
    model.eval()
    out = torch.cat([model(data.image[i:i + chunk].to(DEVICE), data.query[i:i + chunk].to(DEVICE)).cpu()
                     for i in range(0, len(data.query), chunk)])
    model.train()
    return (out > 0).float()


def summarise(values: list[float]) -> str:
    a = np.asarray(values)
    return f"{a.mean():.3f}" if len(a) == 1 else f"{a.mean():.3f} +- {a.std():.3f}"
