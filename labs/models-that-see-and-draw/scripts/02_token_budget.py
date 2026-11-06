"""Count the bits: what a text answer carries, what an image needs, and what a tokenizer keeps.

Recomputes every number in the post's "Count the bits" and "How many tokens an image costs" sections from
the tokenizer settings, and checks them against the values the post prints.
"""

from __future__ import annotations

import math

import _shared  # noqa: F401
from seedraw.common import Report
from seedraw.tokenizers import token_budget

ROWS = [
    # name, height, width, downsampling f, codebook size, numbers the post prints
    ("Chameleon", 512, 512, 16, 8_192, {"tokens": 1024, "max_bits": 13_312, "compression": 470}),
    ("Emu3", 512, 512, 8, 32_768, {"tokens": 4096, "max_bits": 61_440, "compression": 100}),
    ("Janus (generation side)", 384, 384, 16, 16_384, {"tokens": 576}),
    ("This lab's FSQ tokenizer (1 channel)", 16, 16, 4, 1_000, {}),
]


def main() -> None:
    rep = Report("02_token_budget")
    text_bits = math.log2(150_000)
    rep(f"Text: a 150,000-entry vocabulary carries at most log2(150000) = {text_bits:.1f} bits per token.")
    rep(f"A 30-token answer carries at most {30 * text_bits:.0f} bits.  (post: about 17.2 and about 520)")
    raw = 512 * 512 * 3
    rep(f"A 512 x 512 RGB image has {raw:,} values = {raw * 8 / 1e6:.1f} million bits at 8 bits each.  "
        "(post: 786,432 and about 6.3 million)")
    rep()
    rep(f"{'tokenizer':38s} {'tokens':>7s} {'bits/token':>10s} {'max bits':>10s} {'raw bits':>11s} {'compression':>11s}")
    rows, mismatches = [], []
    for name, h, w, f, k, post in ROWS:
        b = token_budget(h, w, f, k, channels=1 if "1 channel" in name else 3)
        rep(f"{name:38s} {b['tokens']:7d} {b['bits_per_token']:10.2f} {b['max_bits']:10.0f} {b['raw_bits']:11,d} "
            f"{b['compression']:10.0f}x")
        for key, val in post.items():
            got = b[key]
            ok = abs(got - val) / val < (0.05 if key == "compression" else 1e-9)  # the post rounds: "about 470", "about 100"
            if not ok:
                mismatches.append(f"{name} {key}: computed {got}, post {val}")
        rows.append({"name": name, "h": h, "w": w, "f": f, "codebook": k, **b})
    rep()
    rep("Every number checked against the post matches." if not mismatches else "MISMATCH: " + "; ".join(mismatches))
    rep.data = {"text_bits_per_token": text_bits, "rows": rows, "mismatches": mismatches}
    rep.save()


if __name__ == "__main__":
    main()
