"""Probe the encoder first: is the answer in the features, and does the VLM use it?

Two BlindTest-style tasks (Rahmanzadehgervi et al., 2024): do two outlined circles overlap, and how many circles are
there. For each, compare the VLM's own answer with linear probes (Alain and Bengio, 2016) on frozen features at
each stage: vision encoder, connector output, and the language model's last position at several layers. A control
probe on shuffled labels (Hewitt and Liang, 2019) checks that the probe cannot just memorise.
"""
import argparse
import random

import _path  # noqa: F401
import torch

from vlmscope import scenes
from vlmscope.intervene import Intervention
from vlmscope.metrics import choice
from vlmscope.model import VLM
from vlmscope.probe import logistic_probe
from vlmscope.report import Report, table

TASKS = {
    "overlap": ("Do the two circles overlap?", ["yes", "no"]),
    "count": ("How many circles are in the image?", ["1", "2", "3", "4"]),
}


def make(task, rng):
    if task == "overlap":
        ov = rng.random() < 0.5
        shapes = scenes.overlap_scene(rng, ov)
        label = 0 if scenes.circles_overlap(shapes) else 1
        return scenes.render(shapes, outline_only=True), label
    n = rng.randint(1, 4)
    return scenes.render(scenes.count_scene(rng, n)), n - 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120, help="images per task")
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()
    rep = Report("06_probe_vs_readout")
    vlm = VLM()
    L = vlm.n_layers
    lm_layers = [0, 7, 15, 22, L - 1]
    out = {}
    for task, (question, cands) in TASKS.items():
        rng = random.Random(args.seed)
        if task == "count":
            # The model answers counts as ' 3': a space token, then a bare digit. Score the digit after the space.
            ids = torch.tensor([vlm.tok(d, add_special_tokens=False).input_ids[0] for d in cands], device=vlm.device)
            space = vlm.tok(" ", add_special_tokens=False).input_ids[0]
        else:
            ids = vlm.word_ids(cands)
        feats = {k: [] for k in ["encoder (mean of 1024 patches)", "connector (mean of 64 tokens)",
                                 *[f"LM layer {l}, last position" for l in lm_layers]]}
        labels, correct = [], 0
        for _ in range(args.n):
            image, y = make(task, rng)
            enc, con = vlm.vision_features(image)
            p = vlm.prompt(image, question)
            if task == "count":
                p = vlm.extend(p, space)
            iv = Intervention(capture_residual=True)
            pick, _ = choice(vlm.run(p, iv), ids)
            correct += int(pick == y)
            labels.append(y)
            feats["encoder (mean of 1024 patches)"].append(enc.mean(0).cpu())
            feats["connector (mean of 64 tokens)"].append(con.mean(0).cpu())
            for l in lm_layers:
                feats[f"LM layer {l}, last position"].append(iv.residual[l][0, p.last].cpu())
        y = torch.tensor(labels)
        n_tr = int(0.6 * args.n)
        majority = float((y[n_tr:] == y[:n_tr].mode().values).float().mean())
        rows = [["VLM answer (forced choice)", f"{100 * correct / args.n:.1f}", ""]]
        res = {"vlm_acc": correct / args.n, "majority_acc": majority, "chance": 1 / len(cands), "probes": {}}
        shuffled = y[torch.randperm(len(y), generator=torch.Generator().manual_seed(0))]
        for name, xs in feats.items():
            x = torch.stack(xs)
            acc = logistic_probe(x[:n_tr], y[:n_tr], x[n_tr:], y[n_tr:], len(cands))
            ctrl = logistic_probe(x[:n_tr], shuffled[:n_tr], x[n_tr:], shuffled[n_tr:], len(cands))
            rows.append([f"probe: {name}", f"{100 * acc:.1f}", f"{100 * ctrl:.1f}"])
            res["probes"][name] = {"acc": acc, "control_acc": ctrl}
        rep(f"\nTask: {task}. '{question}' Labels: {cands}. {args.n} images, probe train/test {n_tr}/{args.n - n_tr}.")
        rep(f"Chance {100 / len(cands):.0f}%, majority class on the test split {100 * majority:.1f}%. The VLM answer is"
            " scored on all images.")
        for line in table(rows, ["readout", "test accuracy %", "shuffled-label control %"]):
            rep(line)
        out[task] = res
    rep.data["tasks"] = out
    rep.save()


if __name__ == "__main__":
    main()
