"""The connector as a bottleneck: how much of the encoder's output survives projection, and where is it lost?

SmolVLM's connector folds each 4 x 4 block of encoder patches into one 12,288-dim vector (pixel shuffle) and maps it
linearly to the language model's 576 dims. Following Li et al. (2025), measure two things per image token:
1. Reconstruction: a ridge map from the projected token back to its 12,288-dim input. Relative error per token.
2. Neighbourhoods: the fraction of each token's 10 nearest neighbours that are the same before and after.
Then ask whether high loss on an object's tokens goes with a less confident answer about that object.
"""
import argparse
import random

import _path  # noqa: F401
import numpy as np
import torch

from vlmscope import scenes
from vlmscope.metrics import choice
from vlmscope.model import VLM
from vlmscope.probe import knn_overlap, ridge, ridge_predict
from vlmscope.report import Report, table


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150, help="scenes; the last third is held out")
    ap.add_argument("--seed", type=int, default=6)
    args = ap.parse_args()
    rep = Report("07_connector")
    vlm = VLM()
    rng = random.Random(args.seed)
    ins, outs, kinds, scene_list = [], [], [], []
    with torch.no_grad():
        for _ in range(args.n):
            shapes = scenes.random_scene(rng, n=2)
            patches, projected = vlm.vision_features(scenes.render(shapes))
            ins.append(vlm.connector.pixel_shuffle(patches[None], vlm.connector.scale_factor)[0].cpu())
            outs.append(projected.cpu())
            inner = scenes.token_mask(shapes[0], min_cover=0.9) | scenes.token_mask(shapes[1], min_cover=0.9)
            touch = scenes.token_mask(shapes[0], min_cover=0.01) | scenes.token_mask(shapes[1], min_cover=0.01)
            kinds.append(np.where(inner, 0, np.where(touch, 1, 2)))  # 0 interior, 1 edge, 2 background
            scene_list.append(shapes)
    n_tr = 2 * args.n // 3
    x = torch.cat(outs)          # [n*64, 576]
    y = torch.cat(ins)           # [n*64, 12288]
    kinds = np.concatenate(kinds)
    tr = slice(0, n_tr * 64)
    te = slice(n_tr * 64, None)
    w = ridge(x[tr], y[tr])
    err_te = ((ridge_predict(w, x[te]) - y[te]) ** 2).sum(-1)
    var_te = ((y[te] - y[tr].mean(0)) ** 2).sum(-1)
    rel = (err_te / var_te).numpy()
    k_te = kinds[n_tr * 64:]
    rep(f"{args.n} scenes ({n_tr} to fit the ridge map, {args.n - n_tr} held out); {64 * (args.n - n_tr)} held-out tokens.")
    rep(f"Projection: {y.shape[1]} -> {x.shape[1]} dims (a linear map, so its rank is at most {x.shape[1]}).")
    rep(f"Held-out R^2 of reconstructing the connector input from its output: {1 - err_te.sum() / var_te.sum():.3f}.")
    rows = []
    names = ["object interior (>=90% covered)", "object edge", "background"]
    for k, name in enumerate(names):
        m = k_te == k
        rows.append([name, int(m.sum()), f"{np.median(rel[m]):.3f}", f"{np.mean(rel[m]):.3f}"])
    for line in table(rows, ["token type", "tokens", "median rel. error", "mean rel. error"]):
        rep(line)
    sub = torch.arange(n_tr * 64, min(len(x), n_tr * 64 + 2000))
    ov = knn_overlap(y[sub], x[sub], k=10)
    rep(f"\nk-NN (k=10, cosine) over {len(sub)} held-out tokens: {100 * float(ov.mean()):.1f}% of neighbours kept, "
        f"so {100 * (1 - float(ov.mean())):.1f}% of neighbour relations changed.")
    # does loss on an object's tokens go with a weaker answer about it?
    colour_ids = vlm.word_ids(list(scenes.COLOURS))
    losses, probs = [], []
    for s_i in range(n_tr, args.n):
        shapes = scene_list[s_i]
        image = scenes.render(shapes)
        r = rel[(s_i - n_tr) * 64:(s_i - n_tr + 1) * 64]
        for target in shapes:
            m = scenes.token_mask(target)
            _, pr = choice(vlm.run(vlm.prompt(image, f"What colour is the {target.kind}?")), colour_ids)
            losses.append(float(r[m].mean()))
            probs.append(float(pr[list(scenes.COLOURS).index(target.colour)]))
    rho = spearman(np.array(losses), np.array(probs))
    acc = np.mean(np.array(probs) > 0.5)
    rep(f"Colour questions on held-out scenes: {len(probs)} prompts, {100 * acc:.1f}% with P(correct) > 0.5.")
    rep(f"Spearman correlation between an object's mean reconstruction error and P(correct colour): {rho:+.3f}.")
    rep.data.update({"r2": float(1 - err_te.sum() / var_te.sum()),
                     "rel_error": {n: {"median": float(np.median(rel[k_te == k])), "mean": float(np.mean(rel[k_te == k]))}
                                   for k, n in enumerate(names)},
                     "knn_kept": float(ov.mean()), "spearman_loss_vs_p": rho, "n_prompts": len(probs)})
    rep.save()


if __name__ == "__main__":
    main()
