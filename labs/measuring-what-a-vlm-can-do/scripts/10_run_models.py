"""Run a real VLM on MMStar or POPE under several protocols and save every raw reply.

MMStar conditions (all greedy, 24 new tokens at most):
  fmt          image, tiling on, prompt ends with the format hint          -> the headline score
  fmt_notile   image, tiling off (one 512 px view)                        -> input settings matter
  bare         image, tiling on, no format hint                           -> the parser matters
  blind        no image, format hint                                      -> S_wv for MMStar's MG / ML
  llm          the matching SmolLM2 Instruct model, no image, format hint -> S_t
  rot1..rot3   as fmt, options rotated by 1, 2, 3 positions                -> CircularEval
POPE: the three settings (random, popular, adversarial), image, tiling on, one-word hint.

Scoring happens in 11_analyse.py, so a new extractor never needs a new GPU run.

    python labs/measuring-what-a-vlm-can-do/scripts/10_run_models.py --model 256m --bench mmstar
    python labs/measuring-what-a-vlm-can-do/scripts/10_run_models.py --model 256m --bench pope
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))

import torch  # noqa: E402

from benchlab.mc import format_prompt, rotate  # noqa: E402
from benchlab.runner import LLM, MODELS, POPE_SETTINGS, VLM, command_line, hardware, load_mmstar, load_pope, pick_device  # noqa: E402

POPE_HINT = "Answer the question using a single word or phrase."


def run_mmstar(args, device) -> dict:
    vlm_id, llm_id = MODELS[args.model]
    rows, skipped = load_mmstar(args.limit)
    items = [r.item for r in rows]
    imgs = [r.image for r in rows]
    out = {"n": len(rows), "skipped": skipped, "index": [r.index for r in rows],
           "category": [r.category for r in rows], "gold": [it.gold_letter for it in items],
           "n_options": [len(it.options) for it in items], "replies": {}, "image_tokens": {}, "seconds": {}}

    def timed(name, fn):
        t = time.time()
        res = fn()
        out["seconds"][name] = round(time.time() - t, 1)
        print(f"{name}: {out['seconds'][name]} s", flush=True)
        return res

    vlm = VLM(vlm_id, device, split=True)
    bs = args.batch_size
    for name, hint, use_img, split, k in [("fmt", True, True, True, 0), ("bare", False, True, True, 0),
                                          ("blind", True, False, True, 0), ("fmt_notile", True, True, False, 0),
                                          ("rot1", True, True, True, 1), ("rot2", True, True, True, 2),
                                          ("rot3", True, True, True, 3)]:
        vlm.split = split
        texts = [format_prompt(rotate(it, k), hint=hint) for it in items]
        replies, tok = timed(name, lambda: vlm.generate(texts, imgs if use_img else None, batch_size=bs, workers=args.workers))
        out["replies"][name], out["image_tokens"][name] = replies, tok
    del vlm
    llm = LLM(llm_id, device)
    out["replies"]["llm"] = timed("llm", lambda: llm.generate([format_prompt(it) for it in items]))
    out["image_tokens"]["llm"] = 0
    return out


def run_pope(args, device) -> dict:
    vlm = VLM(MODELS[args.model][0], device, split=True)
    out = {"replies": {}, "gold": {}, "image_tokens": {}, "seconds": {}}
    for s in POPE_SETTINGS:
        rows = load_pope(s, args.limit)
        t = time.time()
        replies, tok = vlm.generate([f"{r['question']}\n{POPE_HINT}" for r in rows], [r["image"] for r in rows],
                                    max_new_tokens=8, batch_size=args.batch_size, workers=args.workers)
        out["seconds"][s] = round(time.time() - t, 1)
        print(f"{s}: {out['seconds'][s]} s", flush=True)
        out["replies"][s], out["gold"][s], out["image_tokens"][s] = replies, [r["answer"] for r in rows], tok
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(MODELS), default="256m")
    ap.add_argument("--bench", choices=["mmstar", "pope"], default="mmstar")
    ap.add_argument("--limit", type=int, default=None, help="first N items only (smoke runs)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=0, help="CPU worker processes that resize and tile images")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    torch.manual_seed(0)
    device = pick_device()
    t0 = time.time()
    body = run_mmstar(args, device) if args.bench == "mmstar" else run_pope(args, device)
    res = {"hardware": hardware(), "command": command_line(), "model": MODELS[args.model][0],
           "base_llm": MODELS[args.model][1], "limit": args.limit, "total_seconds": round(time.time() - t0, 1), **body}
    suffix = f"_limit{args.limit}" if args.limit else ""
    path = Path(args.out) if args.out else LAB / "results" / "raw" / f"{args.bench}_smolvlm-{args.model}{suffix}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res, ensure_ascii=False, separators=(",", ":")))
    print(f"wrote {path.name} ({path.stat().st_size / 1e3:.0f} kB) in {res['total_seconds']} s")


if __name__ == "__main__":
    main()
