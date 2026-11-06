"""DPO on a real small language model, watching log pi(y_w) and log pi(y_l) separately.

The pair data, tokenisation and log-prob code come from tutorial C6 (c06-post-training/posttrain), which builds
them from scratch. Pairs are C6's synthetic ones: the chosen response is a correct worked solution, the rejected
one is the same text with one number off. Chosen and rejected share all but a few tokens, which is the
high-similarity regime where Razin et al. (2024) report likelihood displacement.
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import torch

C06 = Path(__file__).resolve().parents[3] / "c06-post-training"
if str(C06) not in sys.path:
    sys.path.insert(0, str(C06))

from posttrain.chat import load_tokenizer  # noqa: E402
from posttrain.dpo import collate_pairs, synthetic_pairs, tokenize_pair  # noqa: E402
from posttrain.logprobs import sequence_logprobs  # noqa: E402
from posttrain.models import frozen_copy  # noqa: E402
from posttrain.tasks import split_problems  # noqa: E402

from .losses import dpo_sequence_loss  # noqa: E402

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"


def build_examples(tok, n_pairs: int, seed: int = 1) -> list[dict]:
    train, _ = split_problems(n_pairs, 10, seed=seed)
    return [tokenize_pair(tok, p) for p in synthetic_pairs(train, seed=seed)]


def pair_logps(model, tok, examples: list[dict], device, batch: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    """Summed completion log-probs (chosen, rejected), each (N,). Gradients flow only if the caller enables them."""
    ch, rj = [], []
    for s in range(0, len(examples), batch):
        b = {k: v.to(device) for k, v in collate_pairs(examples[s : s + batch], tok.pad_token_id).items()}
        lp = sequence_logprobs(model, b["input_ids"], b["attention_mask"], b["completion_mask"])
        c, r = lp.chunk(2)
        ch.append(c)
        rj.append(r)
    return torch.cat(ch), torch.cat(rj)


def train(policy, tok, examples: list[dict], device, method: str = "dpo", steps: int = 60, batch: int = 8,
          lr: float = 5e-6, beta: float = 0.1, eval_every: int = 10, n_eval: int = 32, log=print) -> list[dict]:
    """DPO (or DPOP / DPO+NLL) with a cached reference. Every eval_every steps, measure the first n_eval training
    pairs: how far log pi moved from the reference on the chosen and on the rejected responses."""
    ref = frozen_copy(policy)
    with torch.no_grad():
        ref_c, ref_r = pair_logps(ref, tok, examples, device)  # the reference never changes: compute it once
    del ref
    kw = {"dpo": {}, "dpop": {"dpop_lambda": 5.0}, "nll": {"nll_alpha": 0.1}}[method]
    opt = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=0.0)
    rng = random.Random(0)
    history, t0 = [], time.time()
    for step in range(steps + 1):
        if step % eval_every == 0 or step == steps:
            policy.eval()
            with torch.no_grad():
                c, r = pair_logps(policy, tok, examples[:n_eval], device)
            m = beta * ((c - ref_c[:n_eval]) - (r - ref_r[:n_eval]))
            row = {"step": step, "dpo_loss": -torch.nn.functional.logsigmoid(m).mean().item(),
                   "pair_acc": (m > 0).float().mean().item(),
                   "d_logp_chosen": (c - ref_c[:n_eval]).mean().item(),
                   "d_logp_rejected": (r - ref_r[:n_eval]).mean().item(),
                   "frac_chosen_below_ref": (c < ref_c[:n_eval]).float().mean().item(),
                   "sec": time.time() - t0}
            history.append(row)
            log({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()})
            policy.train()
        if step == steps:
            break
        idx = rng.sample(range(len(examples)), batch)
        c, r = pair_logps(policy, tok, [examples[i] for i in idx], device, batch=batch)
        loss = dpo_sequence_loss(c, r, ref_c[idx], ref_r[idx], beta, **kw)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
    return history


def load(device, name: str = MODEL):
    from transformers import AutoModelForCausalLM

    tok = load_tokenizer(name)
    policy = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).to(device)
    return tok, policy
