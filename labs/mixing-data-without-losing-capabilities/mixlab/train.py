"""Training on a mixture: a sampler, two loss normalisations, a loop, and exact-match evaluation.

The sampler is the post's identity made literal. Each row of a batch picks its source with probability p_i, so
the expected gradient is sum_i p_i g_i (under per-example normalisation) or a length-reweighted version of it
(under token normalisation). `weights` may be a fixed dict or a function of the step, which is how the schedule
and the online controllers plug in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping

import torch
import torch.nn.functional as F

from .tasks import IGNORE, PAD, Pool

Weights = Mapping[str, float]


def normalise(w: Weights) -> dict[str, float]:
    total = sum(max(0.0, v) for v in w.values())
    if total <= 0:
        raise ValueError("mixture weights must have a positive sum")
    return {k: max(0.0, v) / total for k, v in w.items()}


def sample_batch(pools: Mapping[str, Pool], weights: Weights, batch: int, gen: torch.Generator):
    """Draw `batch` rows. Row r comes from source i with probability weights[i]. Returns ids, labels, source names."""
    names = [k for k, v in weights.items() if v > 0]
    p = torch.tensor([weights[k] for k in names], dtype=torch.float)
    src = torch.multinomial(p / p.sum(), batch, replacement=True, generator=gen)
    ids, labs, who = [], [], []
    for j, name in enumerate(names):
        n = int((src == j).sum())
        if n == 0:
            continue
        pool = pools[name]
        idx = torch.randint(len(pool), (n,), generator=gen)
        ids.append(pool.input_ids[idx])
        labs.append(pool.labels[idx])
        who += [name] * n
    return torch.cat(ids), torch.cat(labs), who


def token_losses(model, ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Per-position cross-entropy, zero where the label is ignored. Shape (B, T)."""
    logits = model(ids).logits.float()
    ce = F.cross_entropy(logits.transpose(1, 2), labels.clamp(min=0), reduction="none")
    return ce * (labels != IGNORE)


def mixture_loss(model, ids, labels, norm: str = "token") -> torch.Tensor:
    """`token`: mean over all supervised tokens in the batch (what packed LLM trainers do).
    `example`: mean over examples of each example's own token mean (nominal weights come back)."""
    ce = token_losses(model, ids, labels)
    mask = labels != IGNORE
    if norm == "token":
        return ce.sum() / mask.sum()
    if norm == "example":
        return (ce.sum(1) / mask.sum(1).clamp(min=1)).mean()
    raise ValueError(norm)


@torch.no_grad()
def evaluate(model, pools: Mapping[str, Pool], batch: int = 512) -> dict[str, dict[str, float]]:
    """Exact match under teacher forcing: every answer token must be the argmax.

    That is the same as greedy decoding producing the answer, because greedy decoding only differs from teacher
    forcing after its first wrong token. `loss` is the mean per-token loss on the answer.
    """
    model.eval()
    out = {}
    for name, pool in pools.items():
        correct, loss_sum, tok = 0, 0.0, 0
        for s in range(0, len(pool), batch):
            ids, lab = pool.input_ids[s : s + batch], pool.labels[s : s + batch]
            logits = model(ids).logits.float()
            mask = lab != IGNORE
            ok = (logits.argmax(-1) == lab) | ~mask
            correct += int(ok.all(1).sum())
            ce = F.cross_entropy(logits.transpose(1, 2), lab.clamp(min=0), reduction="none") * mask
            loss_sum += float(ce.sum())
            tok += int(mask.sum())
        out[name] = {"acc": correct / len(pool), "loss": loss_sum / tok}
    model.train()
    return out


@torch.no_grad()
def greedy_answers(model, prompts: list[list[int]], max_new: int = 8) -> list[list[int]]:
    """Free generation, no teacher forcing. Stops at the newline token or after max_new tokens."""
    from .tasks import STOI

    model.eval()
    width = max(len(p) for p in prompts)
    # Left-pad is not supported by RoPE positions here, so prompts of one task (same length) go in together.
    assert all(len(p) == width for p in prompts), "greedy_answers expects prompts of equal length"
    x = torch.tensor(prompts)
    outs = [[] for _ in prompts]
    done = torch.zeros(len(prompts), dtype=torch.bool)
    for _ in range(max_new):
        nxt = model(x).logits[:, -1].argmax(-1)
        for r in range(len(prompts)):
            if not done[r]:
                outs[r].append(int(nxt[r]))
        done |= nxt == STOI["\n"]
        if bool(done.all()):
            break
        x = torch.cat([x, nxt[:, None]], 1)
    model.train()
    return outs


@dataclass
class History:
    steps: list[int] = field(default_factory=list)
    evals: list[dict] = field(default_factory=list)
    weights: list[dict] = field(default_factory=list)  # the mixture used at each logged step
    examples_seen: dict[str, int] = field(default_factory=dict)


def train(
    model,
    pools: Mapping[str, Pool],
    weights: Weights | Callable[[int, "History"], Weights],
    steps: int,
    batch: int = 64,
    lr: float = 3e-3,
    seed: int = 0,
    norm: str = "token",
    eval_pools: Mapping[str, Pool] | None = None,
    eval_every: int = 0,
    warmup: int = 50,
    log_weights_every: int = 0,
) -> History:
    """AdamW with a short linear warm-up and a constant rate after it. A new optimiser for every call, so a
    sequence of `train` calls is a sequence of fine-tuning stages."""
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), weight_decay=0.0)
    hist = History()
    model.train()
    for step in range(1, steps + 1):
        w = weights(step, hist) if callable(weights) else weights
        w = normalise(w)
        for g in opt.param_groups:
            g["lr"] = lr * min(1.0, step / max(1, warmup))
        ids, lab, who = sample_batch(pools, w, batch, gen)
        for name in who:
            hist.examples_seen[name] = hist.examples_seen.get(name, 0) + 1
        loss = mixture_loss(model, ids, lab, norm)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if log_weights_every and step % log_weights_every == 0:
            hist.weights.append({"step": step, **w})
        if eval_pools is not None and eval_every and (step % eval_every == 0 or step == steps):
            hist.steps.append(step)
            hist.evals.append(evaluate(model, eval_pools))
    return hist


def scores(ev: Mapping[str, Mapping[str, float]], tasks=None) -> dict[str, float]:
    """Accuracy in percent, per task."""
    tasks = tasks or list(ev)
    return {t: round(100 * ev[t]["acc"], 1) for t in tasks}


def make_pools(tasks, n_train: int = 20000, n_eval: int = 1000, style: str = "", **kw):
    """Train and held-out pools per task. Held-out uses a different seed; overlap is possible for tiny spaces
    (two-digit add has 8,100 distinct prompts), which is why the held-out set measures skill, not novelty."""
    from .tasks import build_pool

    train_p = {t: build_pool(t, n_train, seed=1, style=style, **kw) for t in tasks}
    eval_p = {t: build_pool(t, n_eval, seed=2, style=style, **kw) for t in tasks}
    return train_p, eval_p


__all__ = [
    "History",
    "IGNORE",
    "PAD",
    "evaluate",
    "greedy_answers",
    "make_pools",
    "mixture_loss",
    "normalise",
    "sample_batch",
    "scores",
    "token_losses",
    "train",
]
