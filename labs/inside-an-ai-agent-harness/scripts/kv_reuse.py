"""Section 3 and 5 on a real model: prefill time with and without prefix reuse.

    python labs/inside-an-ai-agent-harness/scripts/kv_reuse.py                       # CPU, SmolLM2-135M
    python labs/inside-an-ai-agent-harness/scripts/kv_reuse.py --device cuda \\
        --model Qwen/Qwen2.5-0.5B-Instruct --dtype bfloat16 --prefix 8192 --delta 1024 --steps 23 \\
        --out kv_reuse_gpu.txt

The context is real text: the C10 harness source, tokenized. Request k is the prefix
plus k chunks, as in an agent turn. Three lanes serve the same requests:

    no reuse        every request is a full forward pass
    reuse           keep the KV cache of the last request and compute only the new suffix
    reuse + edit    the same, but from the middle step on one prefix token is edited in place
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness_lab import C10  # noqa: E402
from harness_lab.kv_reuse import PrefixKV, agent_contexts, full_forward, run_lane  # noqa: E402
from harness_lab.report import Report  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--prefix", type=int, default=1024)
    ap.add_argument("--delta", type=int, default=256)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--out", default="kv_reuse_cpu.txt")
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype)).to(a.device).eval()
    text = "\n".join(p.read_text() for p in sorted((C10 / "harness").rglob("*.py")))
    ids = tok(text)["input_ids"]
    need = a.prefix + a.delta * a.steps
    while len(ids) < need:
        ids = ids + ids
    prefix = ids[:a.prefix]
    chunks = [ids[a.prefix + i * a.delta: a.prefix + (i + 1) * a.delta] for i in range(a.steps)]
    plain = agent_contexts(prefix, chunks)
    edited = agent_contexts(prefix, chunks, edit_at=a.steps // 2 + 1)

    rep = Report(a.out)
    rep(f"model {a.model}, {a.dtype}, device {a.device}")
    rep(f"prefix {a.prefix} tokens, {a.delta} new tokens per step, {a.steps} steps, "
        f"last request {len(plain[-1])} tokens")

    # Exactness: reused-cache logits against a fresh full forward pass, at every step.
    kv, worst, agree = PrefixKV(model, a.device), 0.0, 0
    for ctx in plain + edited[a.steps // 2:]:
        got, _ = kv.forward(ctx)
        ref = full_forward(model, ctx, a.device)
        worst = max(worst, (got - ref).abs().max().item())
        agree += int(got.argmax() == ref.argmax())
    n_checked = a.steps + a.steps - a.steps // 2
    rep(f"exactness: max |logit difference| = {worst:.2e} over {n_checked} requests; "
        f"next-token argmax agrees on {agree} of {n_checked}")

    run_lane(model, plain[:2], "reuse", a.device)          # warm-up, not timed
    lanes = {"no reuse": run_lane(model, plain, "no reuse", a.device),
             "reuse": run_lane(model, plain, "reuse", a.device),
             "reuse + edit": run_lane(model, edited, "reuse", a.device)}
    rep()
    rep(f"{'step':>4} {'tokens':>7} | " + " | ".join(f"{name:>12} computed, ms" for name in lanes))
    for k in range(a.steps):
        row = " | ".join(f"{s[k].computed:>12,} {1000 * s[k].seconds:>12.1f}" for s in lanes.values())
        rep(f"{k + 1:>4} {plain[k].__len__():>7,} | {row}")
    rep()
    base = sum(s.seconds for s in lanes["no reuse"])
    for name, steps in lanes.items():
        secs, comp = sum(s.seconds for s in steps), sum(s.computed for s in steps)
        rep(f"{name:>13}: computed {comp:>9,} tokens, {secs:7.2f} s total, {base / secs:5.1f}x faster than no reuse")
    rep.save()


if __name__ == "__main__":
    main()
