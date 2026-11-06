"""Language priors and visual evidence in a real VLM (SmolVLM-256M-Instruct), on C9's synthetic scenes.

    python labs/aligning-vision-language-models/scripts/prior_split_smolvlm.py --scenes 80 --describe 30

Part A. "What color is the <shape>?" scored over the four palette colours.
    log P(y | x, I) = log P(y | x) + g(y; I | x), with P(y | x) measured on a mid-grey image.
    We report accuracy with the image and with grey, g for the true and the wrong colours, and the KL between the two
    answer distributions. Then we weaken the evidence (blend the image toward grey) and apply visual contrastive
    decoding over the candidates: score = (1 + a) log p(y | I) - a log p(y | I~), with I~ the same image plus noise.
Part B. Image reliance along a long description. Greedy "Describe the image." up to 64 tokens, then per-token
    g_t = log p(y_t | I, y<t) - log p(y_t | grey, y<t), averaged by position. We also check every colour and shape
    word the model writes against the scene, by position.

Downloads the model from the Hugging Face Hub on first use. Reuses C9's scene generator and SmolVLM helpers.
"""

from __future__ import annotations

import argparse
import random
import re
import statistics
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))
sys.path.insert(0, str(LAB.parents[1] / "c09-vlm-post-training"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from groundpref.report import Log  # noqa: E402
from groundpref.train import set_threads  # noqa: E402
from vlmalign.logprobs import build_batch, token_logprobs, forward_logits  # noqa: E402
from vlmalign.model import encode_answer, encode_prompt, generate, load_model, load_processor  # noqa: E402
from vlmalign.scenes import PALETTE, SHAPES, blank_like, render, sample_scene  # noqa: E402

COLOURS = list(PALETTE)
BLENDS = (1.0, 0.3, 0.15, 0.08)
ALPHAS = (0.0, 0.5, 1.0, 2.0, 4.0)


def blend(img: Image.Image, w: float) -> Image.Image:
    """Weak evidence: w = 1 is the image, w = 0 is the grey control."""
    return Image.blend(Image.new("RGB", img.size, (128, 128, 128)), img, w)


def noised(img: Image.Image, sd: float, rng: np.random.Generator) -> Image.Image:
    a = np.asarray(img).astype(np.float32) + rng.normal(0, sd, (img.size[1], img.size[0], 3))
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


@torch.no_grad()
def answer_logps(model, proc, img, question, answers) -> torch.Tensor:
    p = encode_prompt(proc, img, question)
    b = build_batch([p.input_ids[0]] * len(answers), [encode_answer(proc, a) for a in answers],
                    [0] * len(answers), p.pixel_values, proc.tokenizer.pad_token_id)
    lp, _ = token_logprobs(forward_logits(model, b), b.labels)
    return lp.sum(-1)


def part_a(model, proc, n, log):
    rng, nrng = random.Random(0), np.random.default_rng(0)
    answers = [c.capitalize() + "." for c in COLOURS]
    rows, seed = [], 0
    while len(rows) < n:
        seed += 1
        s = sample_scene(seed)
        unique = [o for o in s.objects if s.count(o.shape) == 1]
        if not unique:
            continue
        o = rng.choice(unique)
        q = f"What color is the {o.shape}? Answer with one word."
        img, truth = render(s), COLOURS.index(o.color)
        grey = answer_logps(model, proc, blank_like(s), q, answers)
        row = {"truth": truth, "grey": grey}
        for w in BLENDS:
            weak = blend(img, w)
            row[f"img_{w}"] = answer_logps(model, proc, weak, q, answers)
            row[f"noisy_{w}"] = answer_logps(model, proc, noised(weak, 80.0, nrng), q, answers)
        rows.append(row)
    t = torch.tensor([r["truth"] for r in rows])
    stack = lambda k: torch.stack([r[k] for r in rows])  # noqa: E731
    norm = lambda x: x.log_softmax(-1)  # noqa: E731  renormalise over the four candidates
    lp_i, lp_g = norm(stack("img_1.0")), norm(stack("grey"))
    g = lp_i - lp_g
    true_m = torch.nn.functional.one_hot(t, 4).bool()
    kl = (lp_i.exp() * (lp_i - lp_g)).sum(-1)
    res = {"n": n, "acc_image": (lp_i.argmax(-1) == t).float().mean().item(),
           "acc_grey": (lp_g.argmax(-1) == t).float().mean().item(),
           "grey_argmax_distribution": {COLOURS[i]: (lp_g.argmax(-1) == i).float().mean().item() for i in range(4)},
           "g_true_mean": g[true_m].mean().item(), "g_wrong_mean": g[~true_m].mean().item(),
           "kl_image_vs_grey_mean": kl.mean().item()}
    log(f"Part A. {n} colour questions, candidates {answers}")
    log(f"  accuracy with image {res['acc_image']:.3f}; with grey image (the prior) {res['acc_grey']:.3f}")
    log(f"  grey-image argmax distribution: {res['grey_argmax_distribution']}")
    log(f"  visual information gain g(y; I | x): true colour {res['g_true_mean']:+.2f} nats, "
        f"wrong colours {res['g_wrong_mean']:+.2f} nats; mean KL(P(.|x,I) || P(.|x)) = {res['kl_image_vs_grey_mean']:.2f}")
    log("\n  Weak evidence and contrastive decoding: accuracy over the four colours")
    log("  blend w  " + "  ".join(f"alpha={a:<4g}" for a in ALPHAS))
    res["vcd"] = {}
    for w in BLENDS:
        li, ln = norm(stack(f"img_{w}")), norm(stack(f"noisy_{w}"))
        accs = [(((1 + a) * li - a * ln).argmax(-1) == t).float().mean().item() for a in ALPHAS]
        res["vcd"][str(w)] = dict(zip(map(str, ALPHAS), accs))
        log(f"  {w:<7g}  " + "  ".join(f"{x:<10.3f}" for x in accs))
    return res


def part_b(model, proc, n, log):
    tok = proc.tokenizer
    shape_re = {s: re.compile(rf"\b{s}s?\b", re.I) for s in SHAPES}
    pos_g: dict[int, list[float]] = {}
    mentions: dict[int, list[int]] = {}
    texts = []
    for seed in range(1000, 1000 + n):
        s = sample_scene(seed)
        img, q = render(s), "Describe the image in detail."
        text = generate(model, proc, img, q, max_new_tokens=64)[0].strip()
        texts.append(text)
        p_img, p_grey = encode_prompt(proc, img, q), encode_prompt(proc, blank_like(s), q)
        ids = encode_answer(proc, text)
        lps = []
        for p in (p_img, p_grey):
            b = build_batch([p.input_ids[0]], [ids], [0], p.pixel_values, tok.pad_token_id)
            with torch.no_grad():
                lp, m = token_logprobs(forward_logits(model, b), b.labels)
            lps.append(lp[0][m[0]])
        g = (lps[0] - lps[1]).tolist()
        for t, v in enumerate(g[:64]):
            pos_g.setdefault(t // 16, []).append(v)
        # colour and shape words, checked against the scene, by token position
        colours_in, shapes_in = {o.color for o in s.objects}, {o.shape for o in s.objects}
        offs = tok(" " + text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
        for mt in re.finditer(r"[A-Za-z]+", text):
            w = mt.group(0).lower()
            kind = "colour" if w in COLOURS else ("shape" if any(r.fullmatch(w) for r in shape_re.values()) else None)
            if kind is None:
                continue
            ok = (w in colours_in) if kind == "colour" else (w.rstrip("s") in shapes_in)
            t = next((i for i, (a, b) in enumerate(offs) if a <= mt.start() + 1 < b), 0)
            mentions.setdefault(t // 16, []).append(0 if ok else 1)
    log(f"\nPart B. {n} greedy descriptions, up to 64 tokens. Example: {texts[0]!r}")
    log("  token positions   mean g_t (nats)   tokens   colour/shape mentions   share not in the scene")
    res = {"bins": {}, "examples": texts[:3]}
    for b in sorted(pos_g):
        mm = mentions.get(b, [])
        rate = statistics.mean(mm) if mm else float("nan")
        res["bins"][f"{16 * b}-{16 * b + 15}"] = {"mean_g": statistics.mean(pos_g[b]), "tokens": len(pos_g[b]),
                                                 "mentions": len(mm), "hallucinated_share": rate}
        log(f"  {16 * b:>3d}-{16 * b + 15:<3d}           {statistics.mean(pos_g[b]):+.3f}           "
            f"{len(pos_g[b]):<6d}   {len(mm):<22d}  {rate:.3f}")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=80)
    ap.add_argument("--describe", type=int, default=30)
    args = ap.parse_args()
    set_threads()
    log = Log("prior_split_smolvlm")
    proc = load_processor()
    model = load_model().eval()
    res = {"part_a": part_a(model, proc, args.scenes, log), "part_b": part_b(model, proc, args.describe, log)}
    log.save({"args": vars(args), **res})


if __name__ == "__main__":
    main()
