"""Question 1, getting in: how many visual tokens each input strategy spends on the same picture.

The post's first explainer uses a 1024 x 768 source. We count tokens for it, for a phone photo,
a wide receipt and an A4 page scanned at 150 dpi, under five strategies with their published
default settings. Token counts follow the papers' rules; the tests check them against the
Hugging Face implementations.
"""

import _common  # noqa: F401
from gallery.inputs import fixed_square, internvl_tokens, llava_next_tokens, multicrop_tokens, native_tokens

rep = _common.Report("02_input_tokens")
PINPOINTS = [(336, 672), (672, 336), (672, 672), (1008, 336), (336, 1008)]   # llava-v1.6-mistral-7b-hf

images = {
    "post example 1024x768": (768, 1024),
    "phone photo 4032x3024": (3024, 4032),
    "receipt 600x2000": (2000, 600),
    "A4 page at 150 dpi": (1754, 1240),
}
strategies = {
    "fixed 336 (LLaVA-1.5)": lambda hw: fixed_square(336, 14),
    "anyres tiles (LLaVA-NeXT)": lambda hw: llava_next_tokens(hw, PINPOINTS),
    "448 tiles, <=12 + thumb (InternVL 2.5)": lambda hw: internvl_tokens(hw, max_tiles=12),
    "native, 2x2 merge (Qwen2-VL)": lambda hw: native_tokens(hw),
    "336 crops, 56 px overlap": lambda hw: multicrop_tokens(hw, 336, 56, tokens_per_crop=144),
}
rep("visual tokens per image (the number in brackets is tiles or crops encoded, including any global view)")
rep(f"{'strategy':<40}" + "".join(f"{k:>24}" for k in images))
table = {}
for sname, fn in strategies.items():
    cells, table[sname] = [], {}
    for iname, hw in images.items():
        r = fn(hw)
        table[sname][iname] = r
        cells.append(f"{r['tokens']:>17,} ({r['tiles']:>2})")
    rep(f"{sname:<40}" + "".join(f"{c:>24}" for c in cells))

rep("")
rep("the 'multi-crop' row is a generic overlapping-window rule (Molmo-like crop size, 144 tokens per crop")
rep("after 2x2 pooling of a 24x24 grid), not any one model's exact recipe.")
rep("")
full = internvl_tokens((4000, 4000 * 8 // 5), max_tiles=40)
page40 = internvl_tokens(images["A4 page at 150 dpi"], max_tiles=40)
rep(f"InternVL with max_tiles = 40: an A4 page uses grid {page40['grid']} -> {page40['tokens']:,} tokens; "
    f"the 40-tile maximum (grid {full['grid']}) is {full['tokens']:,} tokens.")
q = native_tokens(images["post example 1024x768"])
rep(f"Qwen2-VL resizes 1024x768 to {q['resized'][1]}x{q['resized'][0]}: {q['tokens']} tokens "
    "(plus 2 vision start/end markers).")
rep.save({"tables": table, "internvl_40_tile_page": page40, "internvl_40_tile_max": full})
