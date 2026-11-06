"""Figure 08: packed rows, document masks and position resets, on two real pretrained models.

Three short documents are packed into one row. For each later document we compare its
per-token loss with the loss of the same document run alone, under four settings. GPT-2 uses
learned absolute positions. SmolLM2 uses rotary positions (RoPE), which only see relative offsets.

A second check moves one document to a large starting position under a document mask, in fp32
and bf16, to see how far RoPE's relative-position property holds in practice.
"""

import argparse

import _common
import torch
from ptlab.packing import SETTINGS, alone_losses, compare, per_token_nll
from transformers import AutoModelForCausalLM, AutoTokenizer

TEXTS = [
    "The cat sat on the mat. It was a sunny day and the cat was happy to stay there all afternoon.",
    "Paris is the capital of France. The city is known for the Eiffel Tower and for its many museums.",
    "def add(a, b):\n    return a + b\n\n\nprint(add(2, 3))\n",
]
MODELS = {"GPT-2 (absolute positions)": "openai-community/gpt2",
          "SmolLM2-135M (RoPE)": "HuggingFaceTB/SmolLM2-135M"}


@torch.no_grad()
def offset_loss(model, doc: list[int], offset: int, dtype) -> float:
    """Mean |loss difference| per token when a document's positions start at `offset` instead of 0."""
    ids = torch.tensor(doc)[None]
    m = model.to(dtype)
    ref = per_token_nll(m(input_ids=ids).logits[0], ids[0])
    pos = torch.arange(offset, offset + ids.shape[1])[None]
    got = per_token_nll(m(input_ids=ids, position_ids=pos).logits[0], ids[0])
    model.float()
    return float((got - ref).abs().max())


def main():
    ap = argparse.ArgumentParser()
    ap.parse_args()
    out = _common.Report("05_packing_masks")
    out("Largest per-token loss difference (nats) from running the document alone. 0 means identical.")
    out("Columns: document 2 and document 3 of the packed row (document 1 is the same in every setting).\n")
    for label, repo in MODELS.items():
        tok = AutoTokenizer.from_pretrained(repo)
        model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.float32, attn_implementation="eager").eval()
        eos = tok.eos_token_id
        docs = [tok(t)["input_ids"] + [eos] for t in TEXTS]
        out(f"{label}: document lengths {[len(d) for d in docs]} tokens, fp32, eager attention")
        res = compare(model, docs, eos)
        for name in SETTINGS:
            out(f"  {name:<42} " + "  ".join(f"{v:9.2e}" for v in res[name]))
        ref = alone_losses(model, docs)
        out(f"  mean loss of documents 2 and 3 alone: {[round(float(r.mean()), 3) for r in ref[1:]]}\n")
        if "RoPE" in label:
            out(f"{label}: one document, positions starting at an offset, document mask only")
            out(f"  {'start position':>15} {'fp32 max diff':>14} {'bf16 max diff':>14}")
            for off in [0, 100, 1000, 8000, 30000]:
                out(f"  {off:>15} {offset_loss(model, docs[1], off, torch.float32):14.2e} "
                    f"{offset_loss(model, docs[1], off, torch.bfloat16):14.2e}")
            out("  Each column compares a shifted run with an unshifted run in the same precision. In fp32 a shift is")
            out("  invisible. In bf16 any shift changes the rounding, and the change is about the same size at 100")
            out("  as at 30,000, so here it is rounding noise, not a drift that grows with position.\n")
    out("Reading: a document mask alone makes packing exact for the RoPE model, because RoPE depends only on")
    out("relative offsets. GPT-2 needs both the mask and the position reset. Resetting positions without a")
    out("mask fixes nothing, because tokens still attend to the previous documents.")
    out.save()


if __name__ == "__main__":
    main()
