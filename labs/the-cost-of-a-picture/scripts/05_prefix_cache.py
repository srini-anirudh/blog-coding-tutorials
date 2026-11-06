"""The prefix-cache trap from "Serving images at scale", on a real model.

Two pictures of the same size become the same run of <|image_pad|> ids. A prefix cache keyed only on token ids
reuses the first picture's KV for the second and answers about the wrong image. Mixing the image hash into
every block that holds image tokens fixes it, and still lets a follow-up question about the same picture skip
the encoder (embedding cache) and most of prefill (prefix cache).
"""

import argparse
import time

import torch
from _common import quiet_transformers, write

from pixelcost import qwen
from pixelcost.serving import EmbeddingCache, PrefixCache, crop_to, image_hash
from pixelcost.synth import solid_shape


def serve(model, processor, image, question, pcache, ecache, device):
    """One request through both caches. Returns the answer and what was skipped."""
    t0 = time.perf_counter()
    inputs = qwen.build_inputs(processor, image, question, device)
    key = image_hash(image, settings="qwen2vl-default")
    feats = ecache.get_or_compute(key, lambda: qwen.encode(model, inputs))
    t1 = time.perf_counter()
    p = qwen.prepare(model, inputs, feats)
    ids = inputs["input_ids"][0].tolist()
    matched, kv = pcache.lookup(ids, model.config.image_token_id, key)
    matched = min(matched, (len(ids) - 1) // pcache.block * pcache.block)   # always run at least one token
    if kv is not None:
        crop_to(kv, matched)
    logits, kv = qwen.prefill(model, p.embeds[:, matched:], p.positions[:, :, matched:], cache=kv)
    t2 = time.perf_counter()
    pcache.insert(ids, model.config.image_token_id, key, kv)
    toks = qwen.decode(model, logits, kv, int(p.positions.max()) + 1, 6, qwen.eos_ids(model))
    return dict(answer=processor.tokenizer.decode(toks, skip_special_tokens=True).strip(), prompt=len(ids),
                reused=matched, encoder_and_prep_ms=(t1 - t0) * 1e3, prefill_ms=(t2 - t1) * 1e3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--side", type=int, default=448)
    args = ap.parse_args()
    quiet_transformers()
    model, processor = qwen.load(device=args.device)
    a = solid_shape("red", args.side, circle=True)
    b = solid_shape("blue", args.side, circle=False)
    q1 = "What color is the shape? Answer with one word."
    q2 = "Is the shape a circle or a square? Answer with one word."
    serve(model, processor, a, q1, PrefixCache(), EmbeddingCache(), args.device)       # warm-up, discarded

    lines, data = [], {}
    for use_hash in (False, True):
        name = "token ids + image hash" if use_hash else "token ids only"
        pc, ec = PrefixCache(use_image_hash=use_hash), EmbeddingCache()
        steps = [("A: red circle, colour?", a, q1), ("B: blue square, colour?", b, q1),
                 ("A again, shape?", a, q2)]
        lines.append(f"Prefix cache keyed on {name}")
        lines.append(f"  {'request':<26} {'answer':<10} {'reused / prompt':>16} {'prep+enc ms':>12} {'prefill ms':>11}")
        out = []
        for label, img, q in steps:
            r = serve(model, processor, img, q, pc, ec, args.device)
            out.append(dict(request=label, **r))
            lines.append(f"  {label:<26} {r['answer']:<10} {r['reused']:>7} / {r['prompt']:<6} "
                         f"{r['encoder_and_prep_ms']:>12.0f} {r['prefill_ms']:>11.0f}")
        lines.append(f"  embedding cache: {ec.hits} hits, {ec.misses} misses\n")
        data[name] = out
    lines.append(f"Pictures {args.side} x {args.side}; both have the same number of visual tokens, so their "
                 "placeholder ids are identical. Blocks of 16 tokens. One run, not a benchmark.")
    tag = "cuda" if args.device.startswith("cuda") else "cpu"
    write(f"05_prefix_cache_{tag}", "\n".join(lines), data, device=args.device)


if __name__ == "__main__":
    main()
