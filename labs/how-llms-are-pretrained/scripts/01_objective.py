"""Figure 01: one sequence, one loss.

Runs GPT-2 (124M) on one sentence and prints, for every position, the true next token, the
probability the model gave it, and -log p. The sum of that column is the sequence loss. It is
checked against the loss Hugging Face computes internally.
"""

import argparse

import _common
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai-community/gpt2")
    ap.add_argument("--text", default="The objective is simple. The distribution being optimised is not.")
    args = ap.parse_args()
    out = _common.Report("01_objective")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).eval()
    ids = torch.tensor(tok(args.text)["input_ids"])
    with torch.no_grad():
        res = model(input_ids=ids[None], labels=ids[None])
    logp = res.logits[0].log_softmax(-1)
    out(f"model {args.model}, text: {args.text!r}, {len(ids)} tokens\n")
    out(f"{'t':>2}  {'prefix ends with':<16} {'true next':<14} {'p(true)':>8} {'-log p':>7}   top guess")
    total = 0.0
    for t in range(len(ids) - 1):
        nxt = int(ids[t + 1])
        nll = -float(logp[t, nxt])
        total += nll
        top = int(logp[t].argmax())
        out(f"{t:>2}  {tok.decode(ids[t])!r:<16} {tok.decode([nxt])!r:<14} {logp[t, nxt].exp():8.4f} {nll:7.3f}   "
            f"{tok.decode([top])!r}")
    n = len(ids) - 1
    out(f"\nsum of -log p over {n} predictions: {total:.4f} nats")
    out(f"mean (what training reports):      {total / n:.4f} nats per token")
    out(f"Hugging Face loss (mean):          {float(res.loss):.4f}")
    out(f"vocabulary size: {logp.shape[-1]}. Each prediction is a distribution over all of it; one entry enters the loss.")
    assert abs(total / n - float(res.loss)) < 1e-4
    out.save()


if __name__ == "__main__":
    main()
