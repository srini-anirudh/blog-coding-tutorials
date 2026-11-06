"""Train the C4 TinyLLM decoder on tool-calling examples, and decode from it with or without a mask.

The objective is the post's: plain next-token cross-entropy, supervised only on the call span.
The prompt (tool list and request) is masked out of the loss with -100.
"""

from __future__ import annotations

import math
import random
import sys
import time
from pathlib import Path
from typing import Callable

import torch

_C04 = Path(__file__).resolve().parents[3] / "c04-llm"
if str(_C04) not in sys.path:
    sys.path.insert(0, str(_C04))

from tinyllm import ModelConfig, TinyLLM  # noqa: E402  (C4's Llama-style decoder, reused as is)

from .world import Example, Tool, Vocab, make_example  # noqa: E402

IGNORE = -100


def build_model(vocab_size: int, hidden: int = 128, layers: int = 4, heads: int = 4, max_len: int = 160) -> TinyLLM:
    cfg = ModelConfig(vocab_size=vocab_size, hidden_size=hidden, num_layers=layers, num_heads=heads,
                      num_kv_heads=heads, intermediate_size=hidden * 3, max_seq_len=max_len)
    return TinyLLM(cfg)


def collate(examples: list[Example], vocab: Vocab) -> tuple[torch.Tensor, torch.Tensor]:
    """Left-aligned batch. Labels are -100 on the prompt and padding, the token id on the call span."""
    rows, labs = [], []
    for ex in examples:
        p, t = vocab.encode(ex.prompt_tokens()), vocab.encode(ex.target_tokens())
        rows.append(p + t)
        labs.append([IGNORE] * len(p) + t)
    width = max(map(len, rows))
    pad = vocab.stoi["<pad>"]
    ids = torch.full((len(rows), width), pad, dtype=torch.long)
    labels = torch.full((len(rows), width), IGNORE, dtype=torch.long)
    for i, (r, lab) in enumerate(zip(rows, labs)):
        ids[i, : len(r)] = torch.tensor(r)
        labels[i, : len(lab)] = torch.tensor(lab)
    return ids, labels


def train(model: TinyLLM, pool: list[Tool], vocab: Vocab, steps: int, batch: int = 64, lr: float = 2e-3,
          seed: int = 0, log_every: int = 200, log: Callable[[str], None] = lambda s: print(s, flush=True),
          device: str = "cpu", max_tools: int = 3) -> list[tuple[int, float]]:
    """Stream fresh requests over a fixed tool pool. Returns (step, loss) pairs."""
    rng = random.Random(seed)
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.98))
    warm = max(1, steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    model.train()
    history, t0, run = [], time.time(), 0.0
    for step in range(1, steps + 1):
        ids, labels = collate([make_example(pool, rng, max_tools=max_tools) for _ in range(batch)], vocab)
        loss = model(ids.to(device), labels=labels.to(device)).loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        run = loss.item() if step == 1 else 0.95 * run + 0.05 * loss.item()
        if step % log_every == 0 or step == steps:
            history.append((step, round(run, 4)))
            log(f"step {step:5d}  loss {run:.4f}  {time.time() - t0:6.1f}s")
    return history


MaskFn = Callable[[list[str]], set[str] | None]


@torch.no_grad()
def decode(model: TinyLLM, vocab: Vocab, ex: Example, max_new: int = 24, mask: MaskFn | None = None,
           temperature: float = 0.0, gen: torch.Generator | None = None) -> list[str]:
    """Greedy (or sampled) decode of the call span. `mask(prefix)` returns the allowed next tokens, or None."""
    model.eval()
    dev = next(model.parameters()).device
    ids = torch.tensor([vocab.encode(ex.prompt_tokens())], device=dev)
    out: list[str] = []
    for _ in range(max_new):
        logits = model(ids).logits[0, -1].float()
        if mask is not None:
            allowed = mask(out)
            if allowed is not None:
                keep = torch.full_like(logits, float("-inf"))
                idx = torch.tensor([vocab.stoi[a] for a in allowed], device=dev)
                keep[idx] = 0.0
                logits = logits + keep  # the one-line mechanism: -inf on every token outside A_t
        if temperature > 0:
            nxt = int(torch.multinomial(torch.softmax(logits / temperature, -1), 1, generator=gen))
        else:
            nxt = int(logits.argmax())
        tok = vocab.itos[nxt]
        out.append(tok)
        if tok == "<eos>":
            break
        ids = torch.cat([ids, torch.tensor([[nxt]], device=dev)], dim=1)
    return out


def save(model: TinyLLM, path: Path, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": {k: v.cpu() for k, v in model.state_dict().items()}, "config": model.config.__dict__, "meta": meta}, path)


def load(path: Path, device: str = "cpu") -> tuple[TinyLLM, dict]:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    model = TinyLLM(ModelConfig(**blob["config"]))
    model.load_state_dict(blob["state"])
    model.to(device).eval()
    return model, blob["meta"]
