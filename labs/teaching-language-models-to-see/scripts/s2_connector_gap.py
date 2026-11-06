"""Sections "Why connecting is not enough" and "The gap between the two models" (Figure 2), measured.

    python labs/teaching-language-models-to-see/scripts/s2_connector_gap.py [--steps 150]

Pretrained SigLIP-base and SmolLM2-135M-Instruct, joined by a fresh 2-layer MLP connector (C7's model).
Only the connector trains, on C7's synthetic shape captions. At each checkpoint we measure the three
mismatches the post names: location (nearest embedding row), scale (norms) and content (captions),
and read out the nearest words to one tracked image token, as Figure 2 does with a sketch.
The trained connector is saved to runs/ for s6 and s7.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from seelab.common import RUNS, Report, pick_device  # noqa: E402
from seelab.gap import image_stats, nearest_rows, object_token, text_reference, text_table  # noqa: E402
from seelab.vlm import EVAL_SEED, batch_of, build, caption_eval, lr_at, to  # noqa: E402
from tinyvlm.data import make_example  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--checkpoints", default="0,10,25,50,100,150")
    ap.add_argument("--eval-n", type=int, default=32)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = pick_device(args.device)
    rep = Report("s2_connector_gap", device)
    model, processor = build("mlp")
    model.set_trainable(vision=False, projector=True, lm=False)
    model.to(device)
    tok = processor.tokenizer
    n_real = processor.image_token_id  # the new <image> id comes right after the original vocabulary
    table = text_table(model.lm, n_real).to(device)
    ref = text_reference(table)
    rep(f"connector: 2-layer MLP, {sum(p.numel() for p in model.projector.parameters()):,} trainable parameters; "
        f"vision and language model frozen. {args.steps} steps x {args.batch_size} pictures, lr {args.lr}")
    rep(f"text embeddings (reference): norm {ref['norm']:.3f}, nearest other row cos {ref['nearest_cos']:.3f}, "
        f"cos to text centroid {ref['cos_to_text_centroid']:.3f}")

    # Fixed probe pictures: their image tokens are measured at every checkpoint.
    probes = [make_example(10_000 + i, seed=EVAL_SEED) for i in range(16)]
    probe_px = processor([processor.apply_chat_template(p["messages"][:1], True) for p in probes],
                         [p["images"] for p in probes])["pixel_values"].to(device)
    tracked = object_token(probes[0]["images"][0])
    rep(f"tracked token: picture 0 ({probes[0]['messages'][1]['content']!r}), token {tracked} of 49 (the cell with most shape pixels)")

    checkpoints = {int(c) for c in args.checkpoints.split(",")}
    rows, opt, seconds = [], torch.optim.AdamW(model.projector.parameters(), lr=args.lr, weight_decay=0.0), 0.0

    def measure(step: int) -> None:
        model.eval()
        with torch.no_grad():
            z = model.get_image_features(probe_px)  # 16 x 49 x 576
        st = image_stats(z.reshape(-1, z.shape[-1]), table)
        _, idx = nearest_rows(z[0, tracked:tracked + 1], table, k=5)
        words = [tok.decode([int(i)]) for i in idx[0]]
        preds, sc = caption_eval(model, processor, args.eval_n, device)
        row = {"step": step, **st, "norm_ratio": st["norm"] / ref["norm"], "nearest_words": words,
               "caption_all_correct": sc["all"], "sample": preds[0], "train_seconds": seconds}
        rows.append(row)
        rep(f"step {step:>4}  norm {st['norm']:7.3f} ({row['norm_ratio']:6.1f}x text)  nearest-row cos {st['nearest_cos']:.3f}  "
            f"cos to text centroid {st['cos_to_text_centroid']:+.3f}  captions all-correct {sc['all']:.2f}  "
            f"nearest words {words}  e.g. {preds[0]!r}")
        model.train()

    if 0 in checkpoints:
        measure(0)
    model.train()
    for step in range(args.steps):
        t0 = time.time()
        batch, _ = batch_of(processor, step * args.batch_size, args.batch_size)
        loss = model(**to(batch, device)).loss
        for gr in opt.param_groups:
            gr["lr"] = lr_at(step, args.steps, args.lr, 10)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.projector.parameters(), 1.0)
        opt.step()
        seconds += time.time() - t0
        if step % 10 == 0:
            print(f"  step {step} loss {loss.item():.3f} {seconds:.0f}s", flush=True)
        if step + 1 in checkpoints:
            measure(step + 1)
    out = RUNS / "s2_connector"
    out.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous().cpu() for k, v in model.projector.state_dict().items()}, out / "projector.safetensors")
    rep(f"training time {seconds:.0f} s (connector forward and backward through the frozen language model)")
    rep("Reading: 'nearest-row cos' near the text reference means image tokens sit where words sit;"
        " a norm ratio far from 1 means the scale differs.")
    rep.save({"reference": ref, "tracked_token": tracked, "rows": rows, "args": vars(args)})


if __name__ == "__main__":
    main()
