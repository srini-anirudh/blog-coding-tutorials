"""The logit lens on image positions, and on the answer position.

p_l(word | i) = softmax(W_U . LN(h_{l,i})). At image positions it asks which word each patch "looks like" to the
language model at layer l (Neo et al., 2024). At the last position it shows the answer forming over depth, and
whether the lowercase meaning appears before the capitalised form the model actually outputs (Zhang et al., 2024).
"""
import argparse
import random

import _path  # noqa: F401
import numpy as np
import torch

from vlmscope import scenes
from vlmscope.intervene import Intervention, intervene
from vlmscope.model import VLM
from vlmscope.report import Report, table

TOPK = 5


def variants(vlm, word):
    """Single-token spellings of a word: ' red', 'red', ' Red', 'Red'."""
    out = set()
    for w in (word, word.capitalize()):
        for s in (" " + w, w):
            ids = vlm.tok(s, add_special_tokens=False).input_ids
            if len(ids) == 1:
                out.add(ids[0])
    return torch.tensor(sorted(out), device=vlm.device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    rep = Report("02_logit_lens")
    vlm = VLM()
    rng = random.Random(args.seed)
    L = vlm.n_layers
    colour_ids = {c: variants(vlm, c) for c in scenes.COLOURS}
    shape_ids = {k: variants(vlm, k) for k in scenes.SHAPES}
    hit_obj = np.zeros(L + 1)         # object token decodes to its own colour (any spelling) in the top 5
    hit_obj_wrong = np.zeros(L + 1)   # object token decodes to some other colour
    hit_obj_shape = np.zeros(L + 1)   # object token decodes to its shape word
    hit_bg = np.zeros(L + 1)          # background token decodes to the object's colour
    n_obj = n_bg = 0
    p_lower = np.zeros(L + 1)
    p_upper = np.zeros(L + 1)
    example = None
    for s_i in range(args.n):
        size = rng.randint(200, 280)
        shape = scenes.Shape(rng.choice(scenes.SHAPES), rng.choice(list(scenes.COLOURS)),
                             rng.randint(0, 512 - size), rng.randint(0, 512 - size), size)
        p = vlm.prompt(scenes.render([shape]), f"What colour is the {shape.kind}?")
        iv = Intervention(capture_residual=True)
        with intervene(iv), torch.no_grad():
            vlm.model(input_ids=p.input_ids, **p.extra, use_cache=False)
        obj = torch.from_numpy(scenes.token_mask(shape, min_cover=0.6)).to(vlm.device)
        bg = torch.from_numpy(~scenes.token_mask(shape, min_cover=0.01)).to(vlm.device)
        n_obj += int(obj.sum())
        n_bg += int(bg.sum())
        own = colour_ids[shape.colour]
        other = torch.cat([v for c, v in colour_ids.items() if c != shape.colour])
        lower = vlm.tok(" " + shape.colour, add_special_tokens=False).input_ids[0]
        upper = vlm.tok(" " + shape.colour.capitalize(), add_special_tokens=False).input_ids[0]
        grids = {}
        for li, layer in enumerate(range(-1, L)):
            with torch.no_grad():
                h = iv.residual[layer][0]
                logits = vlm.lens(h[p.image_pos])                     # [64, V]
                top = logits.topk(TOPK, -1).indices                   # [64, 5]
                last = torch.softmax(vlm.lens(h[p.last]), -1)
            has = lambda ids: torch.isin(top, ids).any(-1)
            hit_obj[li] += float(has(own)[obj].sum())
            hit_obj_wrong[li] += float(has(other)[obj].sum())
            hit_obj_shape[li] += float(has(shape_ids[shape.kind])[obj].sum())
            hit_bg[li] += float(has(own)[bg].sum())
            p_lower[li] += float(last[lower])
            p_upper[li] += float(last[upper])
            if s_i == 0 and layer in (-1, 7, 15, 23, 29):
                grids[layer] = [vlm.tok.decode([t]).strip() or "_" for t in top[:, 0].tolist()]
        if s_i == 0:
            example = (shape, grids, obj.cpu())
    rows = []
    for li, layer in enumerate(range(-1, L)):
        rows.append([("emb" if layer < 0 else layer), f"{100 * hit_obj[li] / n_obj:.1f}",
                     f"{100 * hit_obj_wrong[li] / n_obj:.1f}", f"{100 * hit_obj_shape[li] / n_obj:.1f}",
                     f"{100 * hit_bg[li] / n_bg:.1f}", f"{p_lower[li] / args.n:.3f}", f"{p_upper[li] / args.n:.3f}"])
    rep(f"{args.n} scenes, one large shape each; {n_obj} object tokens (>=60% covered), {n_bg} background tokens.")
    rep(f"Columns 2-5: % of image tokens whose top-{TOPK} logit-lens words include the word (any of ' red', 'red',")
    rep("' Red', 'Red'). Last two columns: mean probability at the last position of the lowercase and the")
    rep("capitalised answer word.")
    for line in table(rows, ["layer", "obj: own colour", "obj: other colour", "obj: shape word", "bg: obj colour",
                             "last P(' red')", "last P(' Red')"]):
        rep(line)
    best = int(np.argmax(hit_obj)) - 1
    rep(f"Peak own-colour decoding at object tokens: layer {best} of {L - 1}, {100 * hit_obj.max() / n_obj:.1f}%.")
    shape, grids, obj = example
    rep(f"\nExample: a {shape.colour} {shape.kind}. Top-1 lens word per image token; [brackets] = object token.")
    for layer, g in grids.items():
        rep(f"layer {'emb' if layer < 0 else layer}:")
        for r in range(8):
            rep("  " + " ".join((f"[{g[r * 8 + c]}]" if obj[r * 8 + c] else g[r * 8 + c])[:10].ljust(10)
                              for c in range(8)))
    rep.data.update({"layers": list(range(-1, L)), "obj_own_colour_pct": (100 * hit_obj / n_obj).tolist(),
                     "obj_other_colour_pct": (100 * hit_obj_wrong / n_obj).tolist(),
                     "obj_shape_pct": (100 * hit_obj_shape / n_obj).tolist(),
                     "bg_obj_colour_pct": (100 * hit_bg / n_bg).tolist(),
                     "last_p_lower": (p_lower / args.n).tolist(), "last_p_upper": (p_upper / args.n).tolist(),
                     "n_obj_tokens": n_obj, "n_bg_tokens": n_bg})
    rep.save()


if __name__ == "__main__":
    main()
