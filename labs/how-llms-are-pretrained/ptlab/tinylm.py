"""A tiny byte-level language model and a training loop that samples from a domain mixture.

The model is Hugging Face `LlamaForCausalLM` with a 256-entry vocabulary (one id per byte), so
no tokenizer is involved. The loop draws each training window from domain i with probability
w_i, and records per-domain validation loss in bits per byte at regular checkpoints.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

SIZES = {  # name -> (hidden, layers, heads, mlp)
    "xs": (64, 2, 2, 192),
    "s": (128, 3, 4, 384),
    "m": (256, 4, 4, 768),
}


def make_model(size: str = "s", seq_len: int = 256, seed: int = 0):
    from transformers import LlamaConfig, LlamaForCausalLM

    h, n_layers, n_heads, mlp = SIZES[size]
    torch.manual_seed(seed)
    cfg = LlamaConfig(
        vocab_size=256, hidden_size=h, num_hidden_layers=n_layers, num_attention_heads=n_heads,
        num_key_value_heads=n_heads, intermediate_size=mlp, max_position_embeddings=seq_len,
        tie_word_embeddings=True, bos_token_id=None, eos_token_id=None, pad_token_id=None,
    )
    cfg._attn_implementation = "sdpa"
    return LlamaForCausalLM(cfg)


def n_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def windows(stream: np.ndarray, starts: np.ndarray, seq_len: int) -> torch.Tensor:
    idx = starts[:, None] + np.arange(seq_len + 1)[None, :]
    return torch.from_numpy(stream[idx].astype(np.int64))


def eval_windows(stream: np.ndarray, n: int, seq_len: int) -> torch.Tensor:
    """n fixed, evenly spaced windows of seq_len + 1 bytes."""
    starts = np.linspace(0, len(stream) - seq_len - 2, n).astype(np.int64)
    return windows(stream, starts, seq_len)


@torch.no_grad()
def window_bpb(model, batch: torch.Tensor, device: str, chunk: int = 64) -> np.ndarray:
    """Bits per byte of each window. (n, L+1) -> (n,)."""
    model.eval()
    out = []
    for i in range(0, len(batch), chunk):
        b = batch[i : i + chunk].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            logits = model(input_ids=b[:, :-1]).logits
        nll = F.cross_entropy(logits.float().transpose(1, 2), b[:, 1:], reduction="none").mean(1)
        out.append(nll.cpu().numpy() / math.log(2))
    model.train()
    return np.concatenate(out)


@dataclass
class RunConfig:
    weights: dict[str, float]
    size: str = "s"
    steps: int = 600
    batch: int = 32
    seq_len: int = 256
    lr: float = 3e-3
    warmup: int = 50
    schedule: str = "cosine"  # or "constant" (after warmup)
    eval_every: int = 100
    last_k: int = 0  # also evaluate at the last k checkpoints, last_every steps apart (for checkpoint noise)
    last_every: int = 10
    seed: int = 0
    unique_bytes: dict[str, int] = field(default_factory=dict)  # cap a domain's training pool


def lr_at(cfg: RunConfig, step: int) -> float:
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / cfg.warmup
    if cfg.schedule == "constant":
        return cfg.lr
    t = (step - cfg.warmup) / max(cfg.steps - cfg.warmup, 1)
    return cfg.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t)))


def train_run(cfg: RunConfig, train: dict[str, np.ndarray], val: dict[str, torch.Tensor], device: str,
              extra_eval=None) -> dict:
    """Train one model on the mixture cfg.weights. val maps domain -> fixed eval windows.

    Returns the config, parameter count, wall time and a list of checkpoints. Each checkpoint has
    the step, bytes seen and per-window bits per byte for every validation set.
    `extra_eval(model) -> dict` adds more numbers to each checkpoint.
    """
    rng = np.random.default_rng(cfg.seed)
    model = make_model(cfg.size, cfg.seq_len, cfg.seed).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.95), weight_decay=0.1)
    names = [d for d, w in cfg.weights.items() if w > 0]
    probs = np.array([cfg.weights[d] for d in names], dtype=float)
    probs /= probs.sum()
    pools = {d: train[d][: cfg.unique_bytes.get(d, len(train[d]))] for d in names}
    eval_steps = set(range(cfg.eval_every, cfg.steps + 1, cfg.eval_every)) | {
        cfg.steps - i * cfg.last_every for i in range(cfg.last_k)}
    ckpts, t0 = [], time.time()
    for step in range(1, cfg.steps + 1):
        counts = rng.multinomial(cfg.batch, probs)
        parts = []
        for d, c in zip(names, counts):
            if c:
                starts = rng.integers(0, len(pools[d]) - cfg.seq_len - 1, size=c)
                parts.append(windows(pools[d], starts, cfg.seq_len))
        b = torch.cat(parts).to(device)
        for g in opt.param_groups:
            g["lr"] = lr_at(cfg, step - 1)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            logits = model(input_ids=b[:, :-1]).logits
        loss = F.cross_entropy(logits.float().reshape(-1, 256), b[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step in eval_steps:
            ck = {"step": step, "bytes_seen": step * cfg.batch * cfg.seq_len, "train_loss": float(loss.detach()),
                  "bpb": {d: window_bpb(model, v, device).tolist() for d, v in val.items()}}
            if extra_eval is not None:
                ck.update(extra_eval(model))
            ckpts.append(ck)
    return {"config": asdict(cfg), "params": n_params(model), "seconds": time.time() - t0, "checkpoints": ckpts,
            "model": model}


def mean_bpb(ck: dict, domain: str, n: int | None = None) -> float:
    v = ck["bpb"][domain]
    return float(np.mean(v if n is None else v[:n]))


@torch.no_grad()
def choice_logprob(model, prompts: list[str], options: list[list[str]], device: str, chunk: int = 512) -> np.ndarray:
    """Sum log-probability of each option's bytes after its prompt. Returns (n_items, n_options)."""
    model.eval()
    seqs, spans = [], []
    for p, opts in zip(prompts, options):
        for o in opts:
            seqs.append(list((p + o).encode()))
            spans.append((len(p.encode()), len(seqs[-1])))
    L = max(map(len, seqs))
    ids = torch.zeros(len(seqs), L, dtype=torch.long)
    for j, s in enumerate(seqs):
        ids[j, : len(s)] = torch.tensor(s)
    flat = np.zeros(len(seqs))
    for i in range(0, len(seqs), chunk):
        b = ids[i : i + chunk].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            logp = model(input_ids=b).logits.float().log_softmax(-1)
        tok_lp = logp[:, :-1].gather(2, b[:, 1:, None])[..., 0].cpu()  # log p of byte t+1
        for j in range(len(b)):
            n_p, n = spans[i + j]
            flat[i + j] = float(tok_lp[j, n_p - 1 : n - 1].sum())
    model.train()
    return flat.reshape(len(prompts), len(options[0]))


def math_choice_items(n: int, seed: int = 123) -> tuple[list[str], list[list[str]], list[int]]:
    """'a + b = ' with four candidate answers: the right one and three that differ in one digit.

    All four have the same number of digits, so a byte-level scorer is not biased towards shorter
    strings. Chance is 25%.
    """
    rng = np.random.default_rng(seed)
    prompts, options, answers = [], [], []
    for _ in range(n):
        a, b = int(rng.integers(0, 1000)), int(rng.integers(0, 1000))
        c = str(a + b)
        opts = [c]
        while len(opts) < 4:
            d = list(c)
            k = int(rng.integers(0, len(d)))
            d[k] = str(int(rng.integers(1 if k == 0 and len(d) > 1 else 0, 10)))
            w = "".join(d)
            if w not in opts:
                opts.append(w)
        perm = rng.permutation(4)
        prompts.append(f"{a} + {b} = ")
        options.append([f"{opts[k]}\n" for k in perm])
        answers.append(int(np.where(perm == 0)[0][0]))
    return prompts, options, answers
