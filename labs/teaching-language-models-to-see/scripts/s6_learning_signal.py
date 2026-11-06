"""Section "Training objectives" and Figure 9: where each loss puts its learning signal on the patch grid.

    python labs/teaching-language-models-to-see/scripts/s6_learning_signal.py      # needs s2's connector

32 scenes, each with two different shapes (one per half). The caption mentions only one of them.
For three objectives we take the gradient with respect to SigLIP's 14 x 14 patch features and ask what share
of it lands on the mentioned shape, the unmentioned shape and the background:
    contrastive   SigLIP's sigmoid loss for the pair (image, "a red circle"), through SigLIP's attention pooling
    next-token    the VLM's caption loss for "One red circle.", through the s2 connector and the frozen LM
    patch loss    regress each patch's own pixels with a random linear decoder (AIMv2- or MAE-style)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from seelab.common import RUNS, Report, pick_device  # noqa: E402
from seelab.signal import (  # noqa: E402
    grad_contrastive, grad_next_token, grad_patch_loss, patch_features, shares, siglip_logits, siglip_text,
    two_object_scene,
)
from seelab.vlm import build  # noqa: E402

PROMPT = "Describe the image."


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--device", default=None)
    ap.add_argument("--connector", default="runs/s2_connector/projector.safetensors", help="relative to the lab folder")
    ap.add_argument("--name", default="s6_learning_signal")
    args = ap.parse_args()
    device = pick_device(args.device)
    rep = Report(args.name, device)
    ckpt = RUNS.parent / args.connector
    if not ckpt.exists():
        sys.exit(f"missing {ckpt.name}: run scripts/s2_connector_gap.py first")
    vlm, processor = build("mlp")
    vlm.projector.load_state_dict(load_file(ckpt))
    vlm.requires_grad_(False)
    vlm.to(device).eval()

    from huggingface_hub import hf_hub_download
    from transformers import SiglipModel

    from tinyvlm.build import VISION_ID

    siglip = SiglipModel.from_pretrained(VISION_ID).to(device).eval()
    siglip.requires_grad_(False)
    vision = siglip.vision_model
    same = all(torch.equal(a, b) for a, b in zip(vision.encoder.parameters(), vlm.vision.encoder.parameters()))
    rep(f"VLM's encoder weights equal SigLIP's: {same}")

    scenes = [two_object_scene(seed) for seed in range(args.n)]
    px = processor.image_processor([s["image"] for s in scenes])["pixel_values"].to(device)
    h = patch_features(vision, px)
    tok_file = hf_hub_download(VISION_ID, "tokenizer.json")
    torch.manual_seed(0)
    decoder = torch.nn.Linear(768, 768).to(device)

    # Sanity check that SigLIP's text side works with our tokenisation: zero-shot over all 18 colour-shape names.
    from tinyvlm.data import COLORS, SHAPES

    names = [(c, s) for c in COLORS for s in SHAPES]
    v_all = siglip_text(siglip, tok_file, [f"a {c} {s}" for c, s in names]).to(device)
    with torch.no_grad():
        top = siglip_logits(siglip, h, v_all).argmax(1).tolist()
    hit = sum(names[t] in sc["objects"] for t, sc in zip(top, scenes)) / args.n
    rep(f"SigLIP check: the best of 18 colour-shape names is one of the scene's two objects in {100 * hit:.0f}% "
        f"of scenes (chance {100 * 2 / 18:.0f}%)")

    table, maps = {}, {}
    for which in (0, 1):  # mention the left object, then the right one
        masks = [[sc["masks"][which], sc["masks"][1 - which]] for sc in scenes]
        v = siglip_text(siglip, tok_file, [f"a {sc['objects'][which][0]} {sc['objects'][which][1]}" for sc in scenes]).to(device)
        msgs = [[{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]},
                 {"role": "assistant", "content": f"One {sc['objects'][which][0]} {sc['objects'][which][1]}."}]
                for sc in scenes]
        batch = processor.collate([processor.build_example(m, [sc["image"]]) for m, sc in zip(msgs, scenes)])
        batch = {k: batch[k].to(device) for k in ("input_ids", "attention_mask", "labels")}
        grads = {"contrastive (SigLIP)": grad_contrastive(siglip, h, v),
                 "next-token (VLM caption)": torch.cat([grad_next_token(vlm, h[i:i + 8], {k: t[i:i + 8] for k, t in batch.items()})
                                                         for i in range(0, args.n, 8)]),
                 "patch loss (own pixels)": grad_patch_loss(h, px, decoder)}
        for name, g in grads.items():
            maps.setdefault(name, []).append(g.norm(dim=-1).detach().cpu())  # n x 196
            per = [shares(g[i], masks[i]) for i in range(args.n)]
            table.setdefault(name, []).extend(per)
        area = [{"obj0": m[0].float().mean().item(), "obj1": m[1].float().mean().item()} for m in masks]
        table.setdefault("area share (reference)", []).extend(
            [{"obj0": a["obj0"], "obj1": a["obj1"], "background": 1 - a["obj0"] - a["obj1"], "max_over_mean": float("nan")}
             for a in area])

    rep(f"\nShare of the per-patch gradient norm, mean over {args.n} scenes x 2 (mention left, mention right)")
    rep(f"{'objective':<28} {'mentioned':>10} {'unmentioned':>12} {'background':>11} {'max/mean':>9}")
    summary = {}
    for name, per in table.items():
        m = {k: sum(p[k] for p in per) / len(per) for k in ("obj0", "obj1", "background", "max_over_mean")}
        summary[name] = m
        rep(f"{name:<28} {m['obj0']:10.3f} {m['obj1']:12.3f} {m['background']:11.3f} {m['max_over_mean']:9.1f}")
    rep("mentioned / unmentioned: the patches covered by the shape the caption names, and by the other shape.")
    rep("max/mean: the strongest patch's gradient over the average patch's; 1 would be perfectly even.")

    # What changes when the caption names the other shape? Only that part of the signal is about the words.
    rep(f"\nSwap which shape the caption names, and compare the two gradient maps of each scene")
    rep(f"{'objective':<28} {'map cosine':>10} {'mentioned > unmentioned':>24} {'change on the shapes':>21} {'shapes area':>12}")
    both = [sc["masks"][0] | sc["masks"][1] for sc in scenes]
    area_both = sum(m.float().mean().item() for m in both) / args.n
    swap = {}
    for name, (m0, m1) in maps.items():
        cos = torch.nn.functional.cosine_similarity(m0, m1, dim=-1).mean().item()
        wins = []
        for i, sc in enumerate(scenes):
            a, b = sc["masks"][0], sc["masks"][1]
            wins += [float(m0[i][a].mean() > m0[i][b].mean()), float(m1[i][b].mean() > m1[i][a].mean())]
        d = (m0 - m1).abs()
        if d.sum() > 0:
            on_shapes = sum((d[i][both[i]].sum() / d[i].sum()).item() for i in range(args.n)) / args.n
            shown = f"{on_shapes:21.3f}"
        else:  # the objective never sees the caption
            on_shapes, shown = None, f"{'no change':>21}"
        swap[name] = {"map_cosine": cos, "mentioned_wins": sum(wins) / len(wins), "change_on_shapes": on_shapes}
        rep(f"{name:<28} {cos:10.3f} {sum(wins) / len(wins):24.2f} {shown} {area_both:12.3f}")
    rep("map cosine: 1 means the gradient map does not depend on which shape the caption names.")
    rep("mentioned > unmentioned: fraction of scene-captions where the named shape's patches get more gradient per patch.")
    rep("change on the shapes: share of |map(name left) - map(name right)| that falls on the two shapes' patches.")
    rep.save({"siglip_sanity_hit_rate": hit, "summary": summary, "swap": swap, "area_both_shapes": area_both,
              "n": args.n})


if __name__ == "__main__":
    main()
