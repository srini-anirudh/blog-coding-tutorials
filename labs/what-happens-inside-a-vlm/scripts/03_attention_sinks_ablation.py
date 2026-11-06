"""Where does the answer position look, layer by layer? Which image tokens draw attention whatever the question?
And does removing them matter, compared with removing the object's own tokens?

Three measurements from the post, made on one small model:
1. The share of the last position's attention that goes to image tokens, per layer (Chen et al., 2024).
2. Visual sinks: image tokens that get the most attention for two different questions about the same image, and
   whether those tokens are high-norm (Kang et al., 2025; Sun et al., 2024).
3. Mean ablation: replace a set of image tokens with the mean image token, at the input to the language model,
   and count how many correct answers break (Neo et al., 2024). Plus FastV-style pruning after layer 2.
"""
import argparse
import random

import _path  # noqa: F401
import numpy as np
import torch

from vlmscope import scenes
from vlmscope.intervene import Intervention
from vlmscope.metrics import choice
from vlmscope.model import VLM
from vlmscope.report import Report, table

FASTV_K = 2  # FastV drops tokens after layer 2


def mean_image_token(vlm, rng, n=16):
    embs = [vlm.vision_features(scenes.render(scenes.random_scene(rng, n=2)))[1] for _ in range(n)]
    return torch.cat(embs).mean(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=2)
    args = ap.parse_args()
    rep = Report("03_attention_sinks_ablation")
    vlm = VLM()
    rng = random.Random(args.seed)
    L, G = vlm.n_layers, vlm.grid ** 2
    mean_tok = mean_image_token(vlm, rng)
    colour_ids = vlm.word_ids(list(scenes.COLOURS))
    share = {g: np.zeros(L) for g in ("prefix", "image", "question", "self")}
    n_runs = 0
    sink_same, sink_on_obj, sink_norm_ratio, top_count = 0, 0, [], 0
    abl_names = ["none", "object + border", "random, same count", "top-attended non-object, same count",
                 "3 highest-norm tokens", f"FastV: lowest-attention 50% after layer {FASTV_K}",
                 f"random 50% after layer {FASTV_K}"]
    abl = {k: {"correct": 0, "p": 0.0, "n": 0, "broken": 0} for k in abl_names}
    dropped_rows = {"fastv": [], "random": []}
    for _ in range(args.n):
        shapes = scenes.random_scene(rng, n=2)
        image = scenes.render(shapes)
        tops = []
        for target in shapes:
            p = vlm.prompt(image, f"What colour is the {target.kind}?")
            gold = list(scenes.COLOURS).index(target.colour)
            iv = Intervention(record_attention=True, capture_residual=True)
            logits = vlm.run(p, iv)
            n_runs += 1
            prefix = torch.arange(0, int(p.image_pos[0]), device=vlm.device)
            for layer in range(L):
                a = iv.attentions[layer][0].mean(0)[p.last]          # [seq], mean over heads
                share["prefix"][layer] += float(a[prefix].sum())
                share["image"][layer] += float(a[p.image_pos].sum())
                share["question"][layer] += float(a[p.question_pos].sum())
                share["self"][layer] += float(a[p.last])
            # attention each image token receives from the last position, layers 2..L-1, mean over heads
            recv = torch.stack([iv.attentions[l][0].mean(0)[p.last, p.image_pos] for l in range(2, L)]).mean(0)
            top3 = recv.topk(3).indices
            tops.append(set(top3.tolist()))
            norms = iv.residual[L // 2][0, p.image_pos].norm(dim=-1)
            sink_norm_ratio.append(float(norms[top3].mean() / norms.median()))
            obj = scenes.dilate(scenes.token_mask(target))
            sink_on_obj += int(sum(obj[i] for i in top3.tolist()))
            top_count += 3
            # ablations ------------------------------------------------------------------------------------
            obj_idx = torch.from_numpy(np.flatnonzero(obj)).to(vlm.device)
            rest = torch.from_numpy(np.flatnonzero(~obj)).to(vlm.device)
            k = min(len(obj_idx), len(rest))
            gen = torch.Generator().manual_seed(1000 + n_runs)
            rand_idx = rest[torch.randperm(len(rest), generator=gen)[:k].to(vlm.device)]
            recv_rest = recv.clone()
            recv_rest[obj_idx] = -1
            top_idx = recv_rest.topk(k).indices
            norm_idx = norms.topk(3).indices
            sets = {"object + border": obj_idx, "random, same count": rand_idx,
                    "top-attended non-object, same count": top_idx, "3 highest-norm tokens": norm_idx}
            results = {"none": logits}
            for name, idx in sets.items():
                pos = p.image_pos[idx]
                results[name] = vlm.run(p, Intervention(embed_patch=(pos, mean_tok.expand(len(pos), -1))))
            # FastV: rank image tokens by the last position's attention at layer K, drop the bottom half for
            # every later layer by blocking all reads from them.
            a_k = iv.attentions[FASTV_K][0].mean(0)[p.last, p.image_pos]
            drop = {"fastv": a_k.topk(G // 2, largest=False).indices,
                    "random": torch.randperm(G, generator=torch.Generator().manual_seed(n_runs))[: G // 2].to(vlm.device)}
            everyone = torch.arange(p.input_ids.shape[1], device=vlm.device)
            for key, name in (("fastv", abl_names[5]), ("random", abl_names[6])):
                src = p.image_pos[drop[key]]
                ko = {l: [(everyone, src)] for l in range(FASTV_K + 1, L)}
                results[name] = vlm.run(p, Intervention(knockout=ko))
                dropped_rows[key].extend((drop[key] // vlm.grid).tolist())
            base_ok = choice(logits, colour_ids)[0] == gold
            for name, lg in results.items():
                pick, probs = choice(lg, colour_ids)
                abl[name]["n"] += 1
                abl[name]["correct"] += int(pick == gold)
                abl[name]["broken"] += int(base_ok and pick != gold)
                abl[name]["p"] += float(torch.softmax(lg, -1)[colour_ids[gold]])
        sink_same += len(tops[0] & tops[1])
    rep(f"{args.n} scenes x 2 colour questions = {n_runs} prompts.")
    rep("\n1. Share of the last position's attention (mean over heads and prompts), by layer.")
    rows = [[l, *(f"{share[g][l] / n_runs:.3f}" for g in share)] for l in range(L)]
    for line in table(rows, ["layer", "prefix (BOS..<global-img>)", "64 image tokens", "question text", "self"]):
        rep(line)
    img_frac = len(p.image_pos) / p.input_ids.shape[1]
    rep(f"Image tokens are {100 * img_frac:.0f}% of the sequence ({len(p.image_pos)} of {p.input_ids.shape[1]}).")
    rep("\n2. Visual sinks: the 3 image tokens that receive the most attention from the last position (layers 2+).")
    rep(f"Shared between the two questions about the same image: {sink_same / (3 * args.n):.2f} of the top 3.")
    rep(f"On the asked-about object (with border): {sink_on_obj / top_count:.2f} of top-3 tokens.")
    rep(f"Their residual norm at layer {L // 2} / median image-token norm: mean {np.mean(sink_norm_ratio):.2f}.")
    rep("\n3. Ablation and pruning. 'broken' = % of prompts answered correctly with no ablation that become wrong.")
    base_ok = abl["none"]["correct"]
    rows = []
    for name in abl_names:
        a = abl[name]
        rows.append([name, f"{100 * a['correct'] / a['n']:.1f}", f"{a['p'] / a['n']:.3f}",
                     f"{100 * a['broken'] / base_ok:.1f}"])
    for line in table(rows, ["tokens removed", "accuracy %", "mean P(correct)", "broken %"]):
        rep(line)
    rep(f"Mean row (0 = top, 7 = bottom) of dropped tokens: FastV {np.mean(dropped_rows['fastv']):.2f}, "
        f"random {np.mean(dropped_rows['random']):.2f}.")
    rep.data.update({"attention_share": {g: (share[g] / n_runs).tolist() for g in share},
                     "sink_shared_frac": sink_same / (3 * args.n), "sink_on_object_frac": sink_on_obj / top_count,
                     "sink_norm_ratio": float(np.mean(sink_norm_ratio)),
                     "ablation": {k: {**v, "acc": v["correct"] / v["n"], "mean_p": v["p"] / v["n"]}
                                  for k, v in abl.items()},
                     "fastv_dropped_mean_row": float(np.mean(dropped_rows["fastv"])),
                     "random_dropped_mean_row": float(np.mean(dropped_rows["random"]))})
    rep.save()


if __name__ == "__main__":
    main()
