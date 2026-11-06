"""The modality gap, and why it only hurts mixed-modality pools.

Part A (from scratch, offline): the two towers of our bi-encoder, at random initialisation and after
training (checkpoint bi64 from script 02). Liang et al.'s first cause of the gap is that each randomly
initialised tower maps all its inputs into a narrow cone of its own.

Part B (real model, downloads openai/clip-vit-base-patch32, about 600 MB): 48 drawn pictures (6 shapes x
8 colours) and their captions. Text queries search (1) a pool of images only, (2) a mixed pool with the
images plus 47 captions of *other* pictures, written in another template. Every text candidate is a wrong
answer, so any text result at rank 1 is modality bias. Then the same pools after removing each
modality's mean vector.

    python labs/searching-with-pixels/scripts/05_modality_gap.py [--skip-clip]
"""

import argparse
import math
import random

import _common as C
import torch
import torch.nn.functional as F

from pixsearch import gap
from pixsearch import retrieve as R
from pixsearch import train as T
from pixsearch.common import Report

ap = argparse.ArgumentParser()
ap.add_argument("--skip-clip", action="store_true")
args = ap.parse_args()

out = Report("05_modality_gap")
result: dict = {}

# ---- Part A
test, queries, rel = C.heldout()
pages = test.images[queries.source]
out("Part A. Our bi-encoder's two towers. Pairs = (query, the page it was written from).")
out(f"{'model':<22}{'cone image':>11}{'cone text':>10}{'gap':>7}{'matched cos':>12}{'unmatched cos':>14}")
for name in ("random init", "trained (bi64)"):
    torch.manual_seed(0)
    model = T.build("bi", 64).eval() if name == "random init" else C.load("bi64")
    with torch.no_grad():
        st = gap.gap_stats(model.encode_pages(pages), model.encode_queries(queries.tokens))
    result[f"scratch_{name}"] = st
    out(f"{name:<22}{st['cone_image']:>11.3f}{st['cone_text']:>10.3f}{st['gap']:>7.3f}{st['matched_cos']:>12.3f}"
        f"{st['unmatched_cos']:>14.3f}")

bi = C.load("bi64")
with torch.no_grad():
    idx = R.index_pages(bi, test.images)
    qv = bi.encode_queries(queries.tokens)
g_vec = idx.mean(0)
dev = gap.offset_shift(qv, idx - g_vec, g_vec)
top1 = ((qv @ idx.T).argmax(1) == (qv @ (idx - g_vec).T).argmax(1)).float().mean().item()
out(f"Subtract the page-side mean g from every page vector: each query's {len(test)} dot-product scores all move by "
    f"the same <q, g> (largest deviation {dev:.1e}, float rounding); top-1 unchanged for {100 * top1:.1f}% of queries.")
result["scratch_offset"] = {"max_deviation": dev, "top1_unchanged_pct": 100 * top1}

if not args.skip_clip:
    from PIL import Image, ImageDraw
    from transformers import CLIPModel, CLIPProcessor

    SHAPES = ["circle", "square", "triangle", "star", "cross", "diamond"]
    COLORS = {"red": (220, 30, 30), "green": (30, 170, 40), "blue": (40, 60, 220), "yellow": (240, 220, 30),
              "purple": (140, 40, 180), "orange": (245, 140, 20), "black": (20, 20, 20), "pink": (245, 130, 190)}

    def draw(shape, rgb, rnd):
        im = Image.new("RGB", (224, 224), tuple(rnd.randint(225, 250) for _ in range(3)))
        d = ImageDraw.Draw(im)
        r = rnd.randint(45, 70)
        cx, cy = rnd.randint(r + 10, 214 - r), rnd.randint(r + 10, 214 - r)
        if shape == "circle":
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=rgb)
        elif shape == "square":
            d.rectangle([cx - r, cy - r, cx + r, cy + r], fill=rgb)
        elif shape == "triangle":
            d.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=rgb)
        elif shape == "diamond":
            d.polygon([(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)], fill=rgb)
        elif shape == "cross":
            w = r // 3
            d.rectangle([cx - w, cy - r, cx + w, cy + r], fill=rgb)
            d.rectangle([cx - r, cy - w, cx + r, cy + w], fill=rgb)
        else:  # star
            pts = [(cx + (r if k % 2 == 0 else r * 0.45) * math.sin(k * math.pi / 5),
                    cy - (r if k % 2 == 0 else r * 0.45) * math.cos(k * math.pi / 5)) for k in range(10)]
            d.polygon(pts, fill=rgb)
        return im

    rnd = random.Random(0)
    combos = [(c, s) for s in SHAPES for c in COLORS]
    images = [draw(s, COLORS[c], rnd) for c, s in combos]
    queries_txt = [f"a photo of a {c} {s}." for c, s in combos]
    cand_txt = [f"a drawing of a {c} {s} shape." for c, s in combos]

    name = "openai/clip-vit-base-patch32"
    clip, proc = CLIPModel.from_pretrained(name).eval(), CLIPProcessor.from_pretrained(name)

    def feats(out_):
        return out_ if torch.is_tensor(out_) else out_.pooler_output

    with torch.no_grad():
        iv = feats(clip.get_image_features(**proc(images=images, return_tensors="pt")))
        qt = feats(clip.get_text_features(**proc(text=queries_txt, return_tensors="pt", padding=True)))
        ct = feats(clip.get_text_features(**proc(text=cand_txt, return_tensors="pt", padding=True)))
    iv, qt, ct = (F.normalize(x, dim=-1) for x in (iv, qt, ct))
    n = len(combos)

    def mixed(q, im, txt):
        """Per query: the pool is all images plus every text candidate except the one with its own content."""
        s_img = q @ im.T
        s_txt = (q @ txt.T).masked_fill(torch.eye(n, dtype=torch.bool), -float("inf"))
        wrong_mod = 100 * (s_txt.max(1).values > s_img.max(1).values).float().mean().item()
        s = torch.cat([s_img, s_txt], dim=1)
        acc = 100 * (s.argmax(1) == torch.arange(n)).float().mean().item()
        return wrong_mod, acc

    st = gap.gap_stats(iv, qt)
    out("")
    out(f"Part B. {name} on {n} drawn pictures (not Flickr30K or COCO).")
    out(f"  cone image {st['cone_image']:.3f}, cone text {st['cone_text']:.3f}, gap {st['gap']:.3f}, "
        f"matched-pair cosine {st['matched_cos']:.3f}, unmatched {st['unmatched_cos']:.3f}")
    out(f"  mean cosine between a query and the text candidates: {(qt @ ct.T).mean():.3f}")
    rows = {}
    for label, (q, im, txt) in {
        "raw CLIP vectors": (qt, iv, ct),
        "each modality's mean removed": (gap.center(qt, torch.cat([qt, ct]).mean(0)), gap.center(iv, iv.mean(0)),
                                         gap.center(ct, torch.cat([qt, ct]).mean(0))),
    }.items():
        img_only = gap.image_pool_top1_accuracy(q, im)
        wm, acc = mixed(q, im, txt)
        st2 = gap.gap_stats(im, q)
        rows[label] = {"image_pool_top1": img_only, "mixed_pool_top1": acc, "wrong_modality_rate": wm,
                       "gap": st2["gap"], "matched_cos": st2["matched_cos"]}
        out(f"  {label}: gap {st2['gap']:.3f}; image-only pool top-1 {img_only:.1f}%; mixed pool top-1 {acc:.1f}%; "
            f"text candidate ranked first for {wm:.1f}% of queries")
    result["clip"] = {"model": name, "n": n, "stats": st, "pools": rows}
out.save(result)
