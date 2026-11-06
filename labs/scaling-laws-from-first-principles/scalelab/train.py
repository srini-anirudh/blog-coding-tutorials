"""Train one model of the ladder for a fixed token budget, then evaluate it.

The optimiser settings follow fixed rules of width and budget, decided before the sweep (section 7's lesson):
- learning rate shrinks with width: 2e-3 * sqrt(256 / d);
- batch size grows with the token budget, about D / 1500 tokens, as a power of two between 16k and 256k;
- warmup is 5% of the run and the cosine decay ends exactly at the last step, so short runs are not handicapped.
None of this was tuned per run. That is a limitation, and the README says so.
"""

from __future__ import annotations

import math
import time

import torch
import torch.nn.functional as F

from scalelab.data import Windows, epoch_order
from scalelab.ladder import Shape, build


def batch_windows(tokens: float, seq: int) -> int:
    target = tokens / 1500
    bt = 2 ** round(math.log2(max(target, 1)))
    return max(16_384, min(262_144, bt)) // seq


def learning_rate(d: int) -> float:
    return 2e-3 * math.sqrt(256 / d)


def lr_at(step: int, steps: int, peak: float, warmup: int) -> float:
    if step < warmup:
        return peak * (step + 1) / warmup
    t = (step - warmup) / max(1, steps - warmup)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t)))


@torch.no_grad()
def evaluate(model, val: Windows, ks=(1, 2, 4, 8, 16), batch: int = 128) -> dict:
    """Mean val loss, next-token accuracy, and 'all k next tokens right' (teacher forced) for each k."""
    model.eval()
    device = val.tokens.device
    loss_sum, n_tok = 0.0, 0
    exact = {k: [0.0, 0] for k in ks}
    for start in range(0, val.count, batch):
        idx = torch.arange(start, min(start + batch, val.count), device=device)
        x, y = val.get(idx)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(x).logits
        logits = logits.float()
        loss_sum += F.cross_entropy(logits.flatten(0, 1), y.flatten(), reduction="sum").item()
        n_tok += y.numel()
        right = (logits.argmax(-1) == y).float()  # (B, T)
        for k in ks:
            # all k of positions t..t+k-1 right: a k-wide min over the row
            run = right.unfold(1, k, 1).amin(-1)
            exact[k][0] += run.sum().item()
            exact[k][1] += run.numel()
    model.train()
    out = {"val_loss": loss_sum / n_tok}
    out.update({f"exact{k}": s / max(c, 1) for k, (s, c) in exact.items()})
    return out


def train_run(shape: Shape, tokens: float, train: Windows, val: Windows, unique: float | None = None,
              seed: int = 0, log_every: int = 0) -> dict:
    """Train `shape` on `tokens` training tokens (drawn from the first `unique` tokens) and evaluate."""
    device = train.tokens.device
    torch.manual_seed(seed)
    model = build(shape).to(device)
    bw = batch_windows(tokens, shape.seq)
    steps = max(1, round(tokens / (bw * shape.seq)))
    n_unique = train.count if unique is None else min(train.count, int(unique // shape.seq))
    order = epoch_order(n_unique, steps * bw, seed, device)
    peak = learning_rate(shape.d)
    warmup = max(1, steps // 20)
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    rest = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": rest, "weight_decay": 0.0}],
                            lr=peak, betas=(0.9, 0.95), fused=device.type == "cuda")
    t0 = time.time()
    loss = torch.zeros((), device=device)
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, steps, peak, warmup)
        x, y = train.get(order[step * bw : (step + 1) * bw])
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(x).logits
        loss = F.cross_entropy(logits.float().flatten(0, 1), y.flatten())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if log_every and step % log_every == 0:
            print(f"  step {step}/{steps} loss {loss.item():.4f}", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    train_s = time.time() - t0
    seen = steps * bw * shape.seq
    res = {
        "shape": shape.name, "d": shape.d, "layers": shape.layers,
        "n_nonembed": shape.non_embedding, "n_total": shape.total,
        "tokens": seen, "unique_tokens": n_unique * shape.seq, "epochs": seen / (n_unique * shape.seq),
        "steps": steps, "batch_tokens": bw * shape.seq, "lr": peak,
        "train_loss_last": loss.item(), "train_seconds": round(train_s, 2),
        "tokens_per_second": round(seen / train_s),
    }
    res.update(evaluate(model, val))
    return res
