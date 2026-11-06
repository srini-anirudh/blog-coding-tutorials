"""Section "Which components are trained" and Figure 4: what each freezing pattern costs, predicted and measured.

    python labs/teaching-language-models-to-see/scripts/s4_freeze_cost.py [--device cuda] [--batch-size 2]

Model: C7's TinyVLM with random weights in the shape of SigLIP-base at 384 px (576 patch tokens),
a 2-layer MLP connector and SmolLM2-135M; 576 image tokens + 100 text tokens per example, as in Figure 4.
For each of the post's four freezing patterns we time forward + backward + AdamW step and count the bytes
of weights, gradients and optimizer state actually held, then compare with the rule of thumb.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from seelab.common import Report, pick_device  # noqa: E402
from seelab.costs import (  # noqa: E402
    PATTERNS, make_batch, param_counts, predicted_bytes, predicted_flops, time_step, tiny_cost_vlm,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--text-tokens", type=int, default=100)
    args = ap.parse_args()
    device = pick_device(args.device)
    tag = "cuda" if device.type == "cuda" else "cpu"
    rep = Report(f"s4_freeze_cost_{tag}", device)
    import transformers

    transformers.logging.set_verbosity_error()
    model = tiny_cost_vlm().to(device)
    params = param_counts(model)
    n_img = model.config.image_seq_len
    tokens = {"vision": n_img, "projector": n_img, "lm": n_img + args.text_tokens}
    rep(f"parameters: vision {params['vision']:,}, connector {params['projector']:,}, LM {params['lm']:,}; "
        f"{n_img} image + {args.text_tokens} text tokens; batch {args.batch_size}; float32; median of {args.repeats}")
    batch = make_batch(model, args.batch_size, args.text_tokens, device)

    with torch.no_grad():  # forward alone, for reference
        model.requires_grad_(False)
        ts = []
        for i in range(args.repeats + 1):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(**batch)
            if device.type == "cuda":
                torch.cuda.synchronize()
            if i:
                ts.append(time.perf_counter() - t0)
    fwd = statistics.median(ts)

    results = {}
    for name, trained in PATTERNS.items():
        results[name] = time_step(model, batch, trained, repeats=args.repeats) | {
            "trained": trained,
            "trained_params": sum(params[p] for p, t in zip(params, trained) if t),
            "pred_flops": predicted_flops(params, tokens, trained) * args.batch_size,
            "pred_bytes_fp32": predicted_bytes(params, trained, 16, 4),
            "pred_bytes_mixed": predicted_bytes(params, trained, 16, 2),
        }
    full = results["joint: train everything"]
    rep(f"\n{'pattern':<40} {'trained params':>15} {'held MB':>8} {'rule MB':>8} {'step s':>7} {'step / full':>11} "
        f"{'rule / full':>11}")
    for name, r in results.items():
        rep(f"{name:<40} {r['trained_params']:>15,} {r['bytes'] / 2**20:8.0f} {r['pred_bytes_fp32'] / 2**20:8.0f} "
            f"{r['seconds']:7.3f} {r['seconds'] / full['seconds']:11.2f} {r['pred_flops'] / full['pred_flops']:11.2f}")
    a = results["align: connector only (LLaVA stage 1)"]
    rep(f"\nforward only (no gradients at all): {fwd:.3f} s = {fwd / full['seconds']:.2f} of a full training step "
        f"(rule: {2 / 6:.2f})")
    rep(f"connector only: {100 * a['trained_params'] / sum(params.values()):.2f}% of the parameters are trained, "
        f"but the step costs {100 * a['seconds'] / full['seconds']:.0f}% of full training (rule {100 * a['pred_flops'] / full['pred_flops']:.0f}%)")
    rep("held MB counts float32 weights + gradients + AdamW moments after one step; the rule here is 16 bytes per trained")
    rep("parameter and 4 per frozen one (float32). Activations are not counted in either column.")
    rep.save({"params": params, "tokens": tokens, "batch_size": args.batch_size, "forward_seconds": fwd,
              "patterns": results})


if __name__ == "__main__":
    main()
