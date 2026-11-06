"""Compare TinyLLM against Hugging Face's LlamaForCausalLM, the trusted implementation.

Part 1 copies our random weights into LlamaForCausalLM and compares logits, loss and greedy output.
Part 2 (--pretrained) goes the other way: it loads SmolLM2-135M from the Hub into TinyLLM.

    python c04-llm/scripts/compare_hf.py
    python c04-llm/scripts/compare_hf.py --pretrained HuggingFaceTB/SmolLM2-135M
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tinyllm import TinyLLM, get_config  # noqa: E402
from tinyllm.hf import from_pretrained, to_hf  # noqa: E402


@torch.no_grad()
def compare_random() -> None:
    torch.manual_seed(0)
    model = TinyLLM(get_config("tiny")).eval()
    hf = to_hf(model)
    ids = torch.randint(0, 256, (2, 128))
    ours, theirs = model(ids, labels=ids), hf(ids, labels=ids)
    print("random tiny model (4.3M params), float32, CPU")
    print(f"  max |logit diff|    {(ours.logits - theirs.logits).abs().max().item():.3e}")
    print(f"  loss ours / HF      {ours.loss.item():.6f} / {theirs.loss.item():.6f}")
    # Same weights, our reference backend: a different order of floating-point operations.
    for layer in model.layers:
        layer.self_attn.backend = "naive"
    naive = model(ids).logits
    print(f"  naive backend diff  {(naive - theirs.logits).abs().max().item():.3e}")
    for layer in model.layers:
        layer.self_attn.backend = "sdpa"
    prompt = ids[:, :8]
    a = model.generate(prompt, max_new_tokens=64)
    b = hf.generate(prompt, attention_mask=torch.ones_like(prompt), max_new_tokens=64, do_sample=False)
    print(f"  64 greedy tokens identical: {torch.equal(a, b)}")


@torch.no_grad()
def compare_pretrained(name: str) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    hf = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()
    model = from_pretrained(name)
    print(f"\n{name} loaded into TinyLLM: {model.num_parameters():,} params, config {model.config}")
    prompt = "The capital of France is"
    ids = tok(prompt, return_tensors="pt").input_ids
    ours, theirs = model(ids).logits, hf(ids).logits
    print(f"  max |logit diff|    {(ours - theirs).abs().max().item():.3e}  (logit scale ~{theirs.abs().max().item():.1f})")
    top = torch.topk(torch.softmax(ours[0, -1], -1), 5)
    print("  next-token top-5:   " + ", ".join(f"{tok.decode([i])!r} {p:.3f}" for p, i in zip(top.values.tolist(), top.indices.tolist())))
    out = model.generate(ids, max_new_tokens=24)
    ref = hf.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=24, do_sample=False)
    print(f"  greedy, ours:       {tok.decode(out[0])!r}")
    print(f"  identical to HF:    {torch.equal(out, ref)}")


def main() -> None:
    from transformers.utils import logging as hf_logging

    hf_logging.disable_progress_bar()
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained", default=None, help="Hub id of a Llama-architecture model")
    args = p.parse_args()
    torch.set_num_threads(2)
    compare_random()
    if args.pretrained:
        compare_pretrained(args.pretrained)


if __name__ == "__main__":
    main()
