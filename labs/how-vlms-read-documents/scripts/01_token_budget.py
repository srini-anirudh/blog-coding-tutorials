"""How many vision tokens an A4 page costs at each resolution, and how many pixels a 10-point letter gets.

Post section: "Resolution and the token budget", Figure 4. No model, only arithmetic and the published resizing rules.
With --check-processor it also runs the real Qwen2-VL image processor (needs the model files in the HF cache).
"""

import argparse

import _common
from docread.budget import (deepseek_mode_tokens, effective_dpi_fixed, effective_dpi_internvl, native_tokens,
                            page_pixels, post_formula_tokens, px_per_em, tiling_tokens_figure, tiling_tokens_internvl)

TEXT_TOKENS = 1000  # the post's assumed content of a dense page
DPIS = [72, 110, 150, 200, 300]

ap = argparse.ArgumentParser()
ap.add_argument("--check-processor", action="store_true")
args = ap.parse_args()
log = _common.Report("01_token_budget")
data = {"rows": []}

log("A4 page (8.27 x 11.69 in). Tokens per page under each input scheme.")
log(f"{'dpi':>4} {'pixels':>12} | {'post formula':>12} {'native (28px)':>13} | {'Fig.4 tiles':>11} {'tokens':>6} |"
    f" {'InternVL max40':>14} {'max12':>6} | {'vision/text':>11}")
for d in DPIS:
    w, h = page_pixels(d)
    nat = native_tokens(d)
    tf, tf_tok = tiling_tokens_figure(d)
    ti, ti_tok = tiling_tokens_internvl(d, max_tiles=40)
    t12, t12_tok = tiling_tokens_internvl(d, max_tiles=12)
    log(f"{d:>4} {round(w):>5} x {round(h):<5} | {post_formula_tokens(d):>12,.0f} {nat:>13,} | {tf:>11} {tf_tok:>6,} |"
        f" {ti:>3} -> {ti_tok:>7,} {t12_tok:>6,} | {nat / TEXT_TOKENS:>10.1f}x")
    data["rows"].append({"dpi": d, "post_formula": round(post_formula_tokens(d)), "native": nat,
                         "figure_tiles": tf, "figure_tiling_tokens": tf_tok, "internvl40_tiles": ti,
                         "internvl40_tokens": ti_tok, "internvl12_tokens": t12_tok})
log("native: Qwen2-VL smart_resize (sides rounded to multiples of 28), one token per 28 x 28 pixels.")
log("Fig.4 tiles: the post's figure covers the page with 448-px tiles at its own size, +1 thumbnail, 256 tokens each.")
log("InternVL: the grid is chosen by aspect ratio alone (closest to 0.707 is 5 x 7 with max 40, 2 x 3 with max 12),")
log("  and the page is resized to that grid. So the tile count does not grow with dpi: low-dpi pages are upscaled.")

log("\nFixed-budget encoder (DeepSeek-OCR native modes): 16-px patches, then 16x token compression.")
for side in (512, 640, 1024, 1280):
    n = deepseek_mode_tokens(side)
    eff = effective_dpi_fixed(side)
    log(f"  {side:>4} px -> {n:>3} tokens, {TEXT_TOKENS / n:>5.1f} text tokens per vision token,"
        f" page shrunk to {eff:5.1f} dpi: 10-pt text gets {px_per_em(10, eff):4.1f} px per em")
    data.setdefault("fixed", []).append({"side": side, "tokens": n, "ratio": TEXT_TOKENS / n, "effective_dpi": eff})

log("\nPixels per em for body text (10 pt) and a footnote (6.5 pt). One Qwen patch is 14 px.")
log(f"{'dpi':>4} {'10 pt':>7} {'6.5 pt':>7} {'patch / em (10 pt)':>19}")
for d in DPIS:
    log(f"{d:>4} {px_per_em(10, d):>7.1f} {px_per_em(6.5, d):>7.1f} {14 / px_per_em(10, d):>19.2f}")
log(f"InternVL 5 x 7 grid: the tiles hold the page at {effective_dpi_internvl(200):.0f} dpi"
    f" (10-pt text {px_per_em(10, effective_dpi_internvl(200)):.1f} px per em). A page rendered below that is"
    " upscaled, which adds no detail.")

if args.check_processor:
    from PIL import Image
    from transformers import AutoImageProcessor

    proc = AutoImageProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    log("\nReal Qwen2-VL image processor (Qwen/Qwen2-VL-2B-Instruct) on a blank A4 page:")
    for d in DPIS:
        w, h = page_pixels(d)
        out = proc(images=Image.new("RGB", (round(w), round(h)), "white"), return_tensors="pt")
        t, gh, gw = out["image_grid_thw"][0].tolist()
        tokens = t * gh * gw // 4
        log(f"  {d:>3} dpi: grid {gh} x {gw} patches -> {tokens:,} tokens (ours: {native_tokens(d):,})")
        data.setdefault("processor", []).append({"dpi": d, "tokens": tokens, "ours": native_tokens(d)})
log.save(data)
