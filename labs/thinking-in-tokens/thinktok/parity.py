"""Parity of n bits, answered directly or after a written running parity.

The post's claim: one forward pass has depth O(L) however long the input is, and every generated token
re-enters at layer 0, so K tokens give depth up to L*K. Parity is the standard example. Asked for the XOR
of n bits in one token, the network must do it inside L layers. Allowed to write the running parity after
each bit, each new token only has to XOR one bit into the token before it.

Two formats over a 3-token vocabulary {0, 1, =}:
  direct  b1 ... bn = y               loss on y only
  cot     b1 ... bn = p1 p2 ... pn    p_i = b1 xor ... xor b_i, loss on every p_i, answer is p_n

Both are trained from scratch on the C4 tutorial's TinyLLM (a Llama-style decoder) with fresh random
strings every step. This is a learnability experiment at small scale. It is consistent with the
circuit-complexity result the post cites, but it does not prove it: a network that fails to learn parity
directly might still be able to represent it.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from . import _paths  # noqa: F401
from tinyllm import ModelConfig, TinyLLM

ZERO, ONE, SEP = 0, 1, 2
VOCAB = 4  # 3 used; 4 keeps the embedding table a power of two


def make_batch(n: int, batch: int, mode: str, generator: torch.Generator | None = None):
    """Return (input_ids, targets, answers). targets is -100 wherever no loss is taken."""
    bits = torch.randint(0, 2, (batch, n), generator=generator)
    running = torch.cumsum(bits, dim=1) % 2  # p_i
    answers = running[:, -1]
    sep = torch.full((batch, 1), SEP)
    if mode == "direct":
        seq = torch.cat([bits, sep, answers[:, None]], dim=1)
    elif mode == "cot":
        seq = torch.cat([bits, sep, running], dim=1)
    else:
        raise ValueError(f"unknown mode {mode!r}")
    inputs, targets = seq[:, :-1], seq[:, 1:].clone()
    targets[:, :n] = -100  # no loss on predicting the input bits or the separator
    return inputs, targets, answers


def answer_len(n: int, mode: str) -> int:
    return 1 if mode == "direct" else n


@dataclass
class ParityConfig:
    n: int = 16
    mode: str = "cot"
    layers: int = 2
    hidden: int = 128
    heads: int = 4
    steps: int = 2000
    batch: int = 256
    lr: float = 1e-3
    eval_size: int = 2000
    seed: int = 0


def build_model(cfg: ParityConfig) -> TinyLLM:
    mc = ModelConfig(
        vocab_size=VOCAB, hidden_size=cfg.hidden, num_layers=cfg.layers, num_heads=cfg.heads,
        num_kv_heads=cfg.heads, intermediate_size=4 * cfg.hidden, max_seq_len=2 * cfg.n + 2,
    )
    return TinyLLM(mc)


def loss_fn(model: TinyLLM, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    logits = model(inputs).logits
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-100)


@torch.no_grad()
def evaluate(model: TinyLLM, cfg: ParityConfig, generator: torch.Generator) -> float:
    """Greedy-decode the answer region and score only the final token against the true parity."""
    inputs, _, answers = make_batch(cfg.n, cfg.eval_size, cfg.mode, generator)
    prompt = inputs[:, : cfg.n + 1]  # bits and '='; the model writes everything after
    out = model.generate(prompt, max_new_tokens=answer_len(cfg.n, cfg.mode), temperature=0.0)
    return (out[:, -1] == answers).float().mean().item()


def train(cfg: ParityConfig, log_every: int = 250, log=print) -> dict:
    torch.manual_seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    eval_gen = torch.Generator().manual_seed(10_000 + cfg.seed)
    model = build_model(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.0)
    history, solved_at = [], None
    t0 = time.perf_counter()
    for step in range(1, cfg.steps + 1):
        model.train()
        inputs, targets, _ = make_batch(cfg.n, cfg.batch, cfg.mode, gen)
        loss = loss_fn(model, inputs, targets)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % log_every == 0 or step == cfg.steps:
            acc = evaluate(model, cfg, torch.Generator().manual_seed(step))
            history.append({"step": step, "loss": round(loss.item(), 4), "acc": round(acc, 4)})
            if solved_at is None and acc >= 0.99:
                solved_at = step
            log(f"  n={cfg.n} {cfg.mode:6s} L={cfg.layers} step {step:5d} loss {loss.item():.4f} answer acc {acc:.3f}")
    final = evaluate(model, cfg, eval_gen)
    return {
        "config": asdict(cfg),
        "params": model.num_parameters(),
        "final_acc": round(final, 4),
        "solved_at_step": solved_at,
        "tokens_written": answer_len(cfg.n, cfg.mode),
        "serial_depth_bound": cfg.layers * answer_len(cfg.n, cfg.mode),
        "seconds": round(time.perf_counter() - t0, 1),
        "history": history,
    }
