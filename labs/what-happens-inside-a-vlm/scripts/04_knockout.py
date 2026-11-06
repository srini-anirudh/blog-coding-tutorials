"""Attention knockout: which routes carry the image to the answer, and at which layers?

At a window of layers, set the attention scores from target positions to source positions to -inf, so the targets
cannot read the sources there (Geva et al., 2023). Slide the window up the network and record the relative change
in P(correct answer), as Zhang et al. (2024) did on LLaVA-1.5 with windows of 9 of 40 layers.
"""
import argparse
import random

import _path  # noqa: F401
import numpy as np
import torch

from vlmscope import scenes
from vlmscope.intervene import Intervention
from vlmscope.metrics import relative_change
from vlmscope.model import VLM
from vlmscope.report import Report, table


def routes(p, obj_tok):
    last = torch.tensor([p.last], device=p.input_ids.device)
    obj = p.image_pos[obj_tok]
    non = p.image_pos[~torch.isin(torch.arange(len(p.image_pos), device=obj_tok.device), obj_tok)]
    return {
        "image -> question": (p.question_pos, p.image_pos),
        "object tokens -> question": (p.question_pos, obj),
        "other image tokens -> question": (p.question_pos, non),
        "question -> last": (last, p.question_pos),
        "image -> last": (last, p.image_pos),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--window", type=int, default=7, help="7 of 30 layers, close to the post's 9 of 40")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()
    rep = Report("04_knockout")
    vlm = VLM()
    rng = random.Random(args.seed)
    L = vlm.n_layers
    starts = list(range(0, L - args.window + 1, args.stride))
    colour_ids = vlm.word_ids(list(scenes.COLOURS))
    effect: dict[str, np.ndarray] = {}
    used = 0
    while used < args.n:
        shapes = scenes.random_scene(rng, n=2)
        target = shapes[0]
        p = vlm.prompt(scenes.render(shapes), f"What colour is the {target.kind}?")
        gold = colour_ids[list(scenes.COLOURS).index(target.colour)]
        logits = vlm.run(p)
        if int(logits[colour_ids].argmax()) != int((colour_ids == gold).nonzero()):
            continue  # knockout asks what a correct answer depends on; skip prompts the model gets wrong
        used += 1
        p0 = float(torch.softmax(logits, -1)[gold])
        obj_tok = torch.from_numpy(np.flatnonzero(scenes.token_mask(target))).to(vlm.device)
        for name, (tgt, src) in routes(p, obj_tok).items():
            row = effect.setdefault(name, np.zeros(len(starts)))
            for j, s in enumerate(starts):
                ko = {l: [(tgt, src)] for l in range(s, s + args.window)}
                pk = float(torch.softmax(vlm.run(p, Intervention(knockout=ko)), -1)[gold])
                row[j] += relative_change(pk, p0)
    rep(f"{args.n} correctly answered colour questions. Window = {args.window} layers, start every {args.stride}.")
    rep("Cells: mean relative change in P(correct colour), in %. -100 = the answer is gone.")
    hdr = ["route \\ window"] + [f"{s}-{s + args.window - 1}" for s in starts]
    rows = [[name] + [f"{100 * v / args.n:+.0f}" for v in row] for name, row in effect.items()]
    for line in table(rows, hdr):
        rep(line)
    rep.data.update({"window": args.window, "starts": starts, "n": args.n,
                     "relative_change_pct": {k: (100 * v / args.n).tolist() for k, v in effect.items()}})
    rep.save()


if __name__ == "__main__":
    main()
