"""Does the model use the image? Score with the image, with a blank image, with no image, and with the base LM.

MMStar's metrics (Chen et al., 2024): MG = S_v - S_wv is what the image adds, ML = max(0, S_wv - S_t) is how much
better the VLM does blind than its own language model. A per-example check follows: how far does the answer
distribution move when the real image is swapped for a blank one?
"""
import argparse
import random

import _path  # noqa: F401
import torch

from vlmscope import scenes
from vlmscope.metrics import choice, multimodal_gain, multimodal_leakage, total_variation
from vlmscope.model import BASE_LM_ID, SUFFIX, VLM
from vlmscope.report import Report, table


def questions(shapes):
    """Three question types. 'Absent animal' is always 'No': a prior can pass it without looking."""
    a, b = shapes
    return [
        ("colour", f"What colour is the {a.kind}?", list(scenes.COLOURS), a.colour),
        ("shape", f"What shape is the {b.colour} object?", list(scenes.SHAPES), b.kind),
        ("absent animal", "Is there a dog in the image?", ["yes", "no"], "no"),
    ]


class BaseLM:
    """The text-only language model SmolVLM-256M was built on, answering the same question with no image."""

    def __init__(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(BASE_LM_ID)
        self.model = AutoModelForCausalLM.from_pretrained(BASE_LM_ID, dtype=torch.float32).eval()

    @torch.no_grad()
    def logits(self, question):
        text = self.tok.apply_chat_template([{"role": "user", "content": question + SUFFIX}], tokenize=False,
                                            add_generation_prompt=True)
        ids = self.tok(text, return_tensors="pt").input_ids
        return self.model(input_ids=ids).logits[0, -1].float()

    def word_ids(self, words):
        return torch.tensor([self.tok(" " + w.capitalize(), add_special_tokens=False).input_ids[0] for w in words])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60, help="number of scenes")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rep = Report("01_blind_baseline")
    vlm, lm = VLM(), BaseLM()
    rng = random.Random(args.seed)
    cache = {}  # blank-image, no-image and base-LM answers depend only on the question text
    stats = {}
    for _ in range(args.n):
        shapes = scenes.random_scene(rng, n=2)
        image = scenes.render(shapes)
        for qtype, q, cands, gold in questions(shapes):
            ids = vlm.word_ids(cands)
            gi = cands.index(gold)
            pick, p_img = choice(vlm.run(vlm.prompt(image, q)), ids)
            if q not in cache:
                _, p_blank = choice(vlm.run(vlm.prompt(scenes.blank(), q)), ids)
                _, p_none = choice(vlm.run(vlm.prompt(None, q)), ids)
                _, p_lm = choice(lm.logits(q), lm.word_ids(cands))
                cache[q] = (p_blank, p_none, p_lm)
            p_blank, p_none, p_lm = cache[q]
            s = stats.setdefault(qtype, {"n": 0, "image": 0, "blank": 0, "none": 0, "lm": 0, "tv": 0.0,
                                         "chance": 1 / len(cands)})
            s["n"] += 1
            s["image"] += pick == gi
            s["blank"] += int(p_blank.argmax()) == gi
            s["none"] += int(p_none.argmax()) == gi
            s["lm"] += int(p_lm.argmax()) == gi
            s["tv"] += total_variation(p_img, p_blank)
    rows, out = [], {}
    tot = {k: 0 for k in ("n", "image", "blank", "none", "lm")}
    for qtype, s in stats.items():
        acc = {k: 100 * s[k] / s["n"] for k in ("image", "blank", "none", "lm")}
        mg = multimodal_gain(acc["image"], acc["none"])
        ml = multimodal_leakage(acc["none"], acc["lm"])
        tv = s["tv"] / s["n"]
        rows.append([qtype, s["n"], f"{100 * s['chance']:.0f}", f"{acc['image']:.1f}", f"{acc['blank']:.1f}",
                     f"{acc['none']:.1f}", f"{acc['lm']:.1f}", f"{mg:+.1f}", f"{ml:.1f}", f"{tv:.2f}"])
        out[qtype] = {**acc, "n": s["n"], "MG": mg, "ML": ml, "mean_tv_image_vs_blank": tv}
        for k in tot:
            tot[k] += s[k]
    acc = {k: 100 * tot[k] / tot["n"] for k in ("image", "blank", "none", "lm")}
    rows.append(["all", tot["n"], "", f"{acc['image']:.1f}", f"{acc['blank']:.1f}", f"{acc['none']:.1f}",
                 f"{acc['lm']:.1f}", f"{multimodal_gain(acc['image'], acc['none']):+.1f}",
                 f"{multimodal_leakage(acc['none'], acc['lm']):.1f}", ""])
    out["all"] = {**acc, "MG": multimodal_gain(acc["image"], acc["none"]),
                  "ML": multimodal_leakage(acc["none"], acc["lm"])}
    rep(f"Forced-choice accuracy (%) over {args.n} synthetic scenes. S_v = with the image, S_wv = the VLM with the")
    rep("image removed from the prompt (MMStar's setting), 'blank' = a plain grey image in its place,")
    rep("S_t = the base LM (SmolLM2-135M-Instruct) on the text alone. TV = mean total-variation distance between")
    rep("the answer distributions with the real and the blank image (0 = the image changed nothing).")
    for line in table(rows, ["question type", "n", "chance", "S_v", "blank", "S_wv", "S_t", "MG", "ML", "TV"]):
        rep(line)
    rep.data["results"] = out
    rep.save()


if __name__ == "__main__":
    main()
