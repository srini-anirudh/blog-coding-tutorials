"""Activation patching on image minimal pairs: where does the colour live, layer by layer?

Clean: the circle is red. Corrupt: the same image with the circle blue. Run the corrupt prompt, copy one group of
residual vectors from the clean run at layer l, and measure
    restored(l, group) = (P_patched(red) - P_corrupt(red)) / (P_clean(red) - P_corrupt(red)).
Near 1 means that group, at that layer, is enough to bring the clean answer back (Meng et al., 2022).
"""
import argparse
import random

import _path  # noqa: F401
import numpy as np
import torch

from vlmscope import scenes
from vlmscope.intervene import Intervention
from vlmscope.metrics import restoration
from vlmscope.model import VLM
from vlmscope.report import Report, table


def groups(p, kind, obj_tok):
    word = p.find(kind)
    obj = p.image_pos[obj_tok]
    other = p.image_pos[~torch.isin(torch.arange(len(p.image_pos), device=obj_tok.device), obj_tok)]
    q_rest = p.question_pos[p.question_pos != word]
    return {"object image tokens": obj, "other image tokens": other, "object word":
            torch.tensor([word], device=p.input_ids.device),
            "other text tokens": q_rest, "last position": torch.tensor([p.last], device=p.input_ids.device)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=4)
    args = ap.parse_args()
    rep = Report("05_patching")
    vlm = VLM()
    rng = random.Random(args.seed)
    L = vlm.n_layers
    layers = [-1] + list(range(0, L, 2)) + ([L - 1] if (L - 1) % 2 else [])
    colour_ids = vlm.word_ids(list(scenes.COLOURS))
    total: dict[str, np.ndarray] = {}
    used, gaps = 0, []
    while used < args.n:
        shapes = scenes.random_scene(rng, n=2)
        new = rng.choice([c for c in scenes.COLOURS if c not in {s.colour for s in shapes}])
        q = f"What colour is the {shapes[0].kind}?"
        clean = vlm.prompt(scenes.render(shapes), q)
        corrupt = vlm.prompt(scenes.render(scenes.recolour(shapes, 0, new)), q)
        gold = colour_ids[list(scenes.COLOURS).index(shapes[0].colour)]
        iv = Intervention(capture_residual=True)
        p_clean = float(torch.softmax(vlm.run(clean, iv), -1)[gold])
        p_corrupt = float(torch.softmax(vlm.run(corrupt), -1)[gold])
        if p_clean - p_corrupt < 0.5:
            continue  # keep pairs where the edit really flips the answer
        used += 1
        gaps.append(p_clean - p_corrupt)
        obj_tok = torch.from_numpy(np.flatnonzero(scenes.token_mask(shapes[0]))).to(vlm.device)
        for name, pos in groups(corrupt, shapes[0].kind, obj_tok).items():
            row = total.setdefault(name, np.zeros(len(layers)))
            for j, layer in enumerate(layers):
                rep_vec = iv.residual[layer][0, pos]
                patch = Intervention(embed_patch=(pos, rep_vec)) if layer < 0 else \
                    Intervention(patch={layer: (pos, rep_vec)})
                pp = float(torch.softmax(vlm.run(corrupt, patch), -1)[gold])
                row[j] += restoration(pp, p_corrupt, p_clean)
    rep(f"{args.n} minimal pairs (one shape recoloured); mean P_clean - P_corrupt = {np.mean(gaps):.2f}.")
    rep("Cells: mean restoration, 0 = no effect, 1 = clean answer fully restored. 'emb' = input embeddings.")
    hdr = ["patched group \\ layer"] + [("emb" if l < 0 else str(l)) for l in layers]
    rows = [[name] + [f"{v / args.n:.2f}" for v in row] for name, row in total.items()]
    for line in table(rows, hdr):
        rep(line)
    rep.data.update({"layers": layers, "n": args.n,
                     "restoration": {k: (v / args.n).tolist() for k, v in total.items()}})
    rep.save()


if __name__ == "__main__":
    main()
