"""Small text dissolves: train a CRNN + CTC line reader on rendered text, then measure how accuracy falls with the
number of pixels per em. Then read the same lines the two ways a page parser can: from a page shrunk to a small
input, or from a full-resolution crop.

Post sections: "The pipeline tradition" (CRNN, CTC), "Resolution and the token budget" (Figure 4), and the two-pass
design in "Specialist document parsers" (MinerU2.5 reads layout on a 1036-px thumbnail and content on native crops).
"""

import argparse
import random
import time

import torch

import _common
from docread.budget import A4_INCHES, px_per_em
from docread.metrics import levenshtein
from docread.ocr import downscale, random_text, read_lines, render_line, train_crnn, CRNN

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=3000)
ap.add_argument("--n-eval", type=int, default=400, help="lines per condition")
ap.add_argument("--load", action="store_true", help="reuse runs/crnn.pt instead of training")
args = ap.parse_args()
log = _common.Report("02_small_text")
ckpt = _common.LAB / "runs" / "crnn.pt"

if args.load and ckpt.exists():
    model = CRNN()
    model.load_state_dict(torch.load(ckpt))
    model.eval()
    log(f"loaded {ckpt.relative_to(_common.LAB)}")
else:
    t0 = time.time()
    model = train_crnn(steps=args.steps, log=log)
    log(f"trained {args.steps} steps (batch 64, em sizes 3 to 28 px) in {time.time() - t0:.0f} s")
    ckpt.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), ckpt)


def score(images, texts):
    preds = []
    for i in range(0, len(images), 200):
        preds += read_lines(model, images[i:i + 200])
    errs = sum(levenshtein(p, t) for p, t in zip(preds, texts))
    amounts = [(p, t) for p, t in zip(preds, texts) if t[0].isdigit()]
    return {"char_acc": 1 - errs / sum(len(t) for t in texts),
            "line_exact": sum(p == t for p, t in zip(preds, texts)) / len(texts),
            "amount_exact": sum(p == t for p, t in amounts) / max(1, len(amounts)),
            "examples": [(t, p) for p, t in list(zip(preds, texts))[:3]]}


def eval_texts(seed):
    rng = random.Random(seed)
    return [random_text(rng) for _ in range(args.n_eval)]


data = {"steps": args.steps, "by_em": [], "by_dpi": [], "two_pass": []}
log(f"\nAccuracy by pixels per em ({args.n_eval} held-out lines each; amounts like 1,580.00, words, codes):")
log(f"{'px/em':>6} {'char acc':>9} {'line exact':>11} {'amount exact':>13}  example (truth -> read)")
for em in (3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 28):
    texts = eval_texts(1000 + em)
    s = score([render_line(t, em) for t in texts], texts)
    log(f"{em:>6} {s['char_acc']:>9.3f} {s['line_exact']:>11.3f} {s['amount_exact']:>13.3f}  "
        f"{s['examples'][0][0]} -> {s['examples'][0][1]}")
    data["by_em"].append({"px_per_em": em, **{k: v for k, v in s.items() if k != "examples"}})

log("\nThe same reader on 10-pt body text and a 6.5-pt footnote, rendered at each dpi:")
log(f"{'dpi':>4} {'10 pt px/em':>12} {'line exact':>11} {'6.5 pt px/em':>13} {'line exact':>11}")
for d in (72, 110, 150, 200, 300):
    row = {"dpi": d}
    cells = []
    for pt in (10, 6.5):
        texts = eval_texts(2000 + d + int(pt * 10))
        em = px_per_em(pt, d)
        s = score([render_line(t, em) for t in texts], texts)
        row[f"{pt}pt"] = {"px_per_em": em, "line_exact": s["line_exact"], "char_acc": s["char_acc"]}
        cells.append(f"{em:>12.1f} {s['line_exact']:>11.3f}")
    log(f"{d:>4} {cells[0]} {cells[1].rjust(25)}")
    data["by_dpi"].append(row)

log("\nOne pass on a shrunk page versus a native-resolution crop of each element (page rendered at 200 dpi):")
page_long = A4_INCHES[1] * 200
log(f"{'input':<44} {'scale':>6} {'10 pt exact':>12} {'6.5 pt exact':>13}")
for name, side in (("native crop (two-pass, stage 2)", None), ("whole page at 1280 px (DeepSeek-OCR Large)", 1280),
                   ("whole page at 1036 px (MinerU2.5 layout view)", 1036), ("whole page at 1024 px (fixed 256 tokens)", 1024),
                   ("whole page at 640 px (fixed 100 tokens)", 640)):
    factor = 1.0 if side is None else side / page_long
    out = {"input": name, "scale": factor}
    for pt in (10, 6.5):
        texts = eval_texts(3000 + int(pt * 10))
        imgs = [render_line(t, px_per_em(pt, 200)) for t in texts]
        if side is not None:
            imgs = [downscale(im, factor) for im in imgs]
        out[f"{pt}pt"] = score(imgs, texts)["line_exact"]
    log(f"{name:<44} {factor:>6.3f} {out['10pt']:>12.3f} {out['6.5pt']:>13.3f}")
    data["two_pass"].append(out)
log("Every line here is read by the same small CRNN; only the pixels it gets change.")
log.save(data)
