"""Print one preference pair per hallucination kind and the tensors of one batch.

    python scripts/show_pairs.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from vlmalign.dpo import pair_batch  # noqa: E402
from vlmalign.logprobs import IGNORE  # noqa: E402
from vlmalign.model import load_processor  # noqa: E402
from vlmalign.preference import build_pairs  # noqa: E402
from vlmalign.tasks import KINDS  # noqa: E402


def main() -> None:
    torch.set_num_threads(2)
    for mode in ("balanced", "biased"):
        print(f"--- {mode} ---")
        for p in build_pairs(len(KINDS), mode, seed=11):
            print(f"{p.kind:8s} {p.prompt}")
            print(f"{'':8s}   chosen   {p.chosen}")
            print(f"{'':8s}   rejected {p.rejected}")

    proc = load_processor()
    pairs = build_pairs(2, "balanced", seed=11)
    b = pair_batch(proc, pairs)
    print("\n--- one batch: 2 pairs, so 4 rows and 2 images ---")
    for name in ("input_ids", "attention_mask", "labels", "image_index", "pixel_values"):
        t = getattr(b, name)
        print(f"{name:15s} {t.shape}  dtype={t.dtype}")
    image_id = proc.tokenizer.convert_tokens_to_ids("<image>")
    print(f"image tokens in row 0: {(b.input_ids[0] == image_id).sum().item()}")

    row = 0
    n_img = (b.input_ids[row] == image_id).sum().item()
    print(f"\n--- row {row}, with the {n_img} <image> ids collapsed ---")
    seen = False
    for tid, lab, att in zip(b.input_ids[row].tolist(), b.labels[row].tolist(), b.attention_mask[row].tolist()):
        if tid == image_id:
            if not seen:
                print(f"{'<image> x ' + str(n_img):24s} mask=1 label={IGNORE}")
                seen = True
            continue
        print(f"{proc.tokenizer.decode([tid])!r:24s} id={tid:<6d} mask={att} label={lab}")


if __name__ == "__main__":
    main()
