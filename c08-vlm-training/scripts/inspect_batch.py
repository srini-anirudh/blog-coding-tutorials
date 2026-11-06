"""Build one mixed batch and print what is inside its tensors. No model weights needed.

    python c08-vlm-training/scripts/inspect_batch.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vlmtrain.data import TASKS, make_example, text_sample  # noqa: E402
from vlmtrain.processor import IGNORE, TinyVLMProcessor  # noqa: E402
from vlmtrain.replay import FALLBACK_PASSAGES  # noqa: E402
from vlmtrain.tokenizer import build_tokenizer  # noqa: E402


def main() -> None:
    tok = build_tokenizer()
    proc = TinyVLMProcessor(tok, tokens_per_tile=49, tile=224, strategy="tile", max_tiles=4)
    samples = [make_example(t, 0) for t in TASKS] + [text_sample(FALLBACK_PASSAGES[3])]
    print(f"{'task':<11}{'image':>11}{'tiles':>7}{'visual':>8}{'text':>6}{'target':>8}  answer")
    for s in samples:
        e = proc.encode(s)
        n_vis = e.input_ids.count(proc.image_token_id)
        n_sup = sum(l != IGNORE for l in e.labels)
        size = "x".join(map(str, s["images"][0].size)) if s["images"] else "-"
        ans = s["meta"].get("answer", "(all tokens)")
        print(f"{s['task']:<11}{size:>11}{sum(e.tiles_per_image):>7}{n_vis:>8}{len(e.input_ids) - n_vis:>6}{n_sup:>8}  {ans}")

    b = proc(samples)
    print("\ninput_ids     ", tuple(b["input_ids"].shape))
    print("attention_mask", tuple(b["attention_mask"].shape), f"padding share {1 - b['attention_mask'].float().mean():.1%}")
    print("labels        ", tuple(b["labels"].shape), f"supervised share {(b['labels'] != IGNORE).float().mean():.1%}")
    print("pixel_values  ", tuple(b["pixel_values"].shape))

    e = proc.encode(samples[1])
    print("\nOne VQA sample, token by token (run-length encoded):")
    run = None
    for tid, lab in list(zip(e.input_ids, e.labels)) + [(None, None)]:
        name = "<image>" if tid == proc.image_token_id else (tok.decode([tid]) if tid is not None else None)
        key = (name, lab != IGNORE) if name == "<image>" else None
        if run and run[0] == key and key is not None:
            run[1] += 1
            continue
        if run:
            print(f"  {repr(run[2]):<14} x{run[1]:<3} {'target' if run[3] else '-100'}")
        if tid is not None:
            run = [key, 1, name, lab != IGNORE]


if __name__ == "__main__":
    main()
