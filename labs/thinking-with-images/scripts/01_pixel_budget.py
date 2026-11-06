"""Where the pixels go: the post's resolution arithmetic, computed, plus what the plaque looks like at each width.

Reproduces every number in "What reasoning tokens cannot recover" and "How each axis scales". CPU only, about 5 s.
    python labs/thinking-with-images/scripts/01_pixel_budget.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))
from lookagain.budget import Encoding, attention_cost, image_share, min_width_for_text, resize_long_side  # noqa: E402
from lookagain.scene import GRID, TEXT_H, H, W, cell_box, make_scene, resize_to  # noqa: E402
from lookagain.util import header  # noqa: E402
from lookagain.views import View, detail_retained  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(LAB / "results" / "01_pixel_budget"))
    args = ap.parse_args()
    lines = [header("python labs/thinking-with-images/scripts/01_pixel_budget.py"), ""]
    say = lines.append

    full = Encoding(W, H, W, H)
    say(f"Full photo {W} x {H}, 14 px patches, no merging: grid {full.grid}, {full.patches:,} visual tokens")
    ew, eh = resize_long_side(W, H, 448)
    small = Encoding(W, H, ew, eh)
    say(f"Resized to {ew} x {eh}: grid {small.grid}, {small.tokens} tokens; "
        f"text height {TEXT_H} px -> {small.text_height(TEXT_H):.2f} px")
    need = min_width_for_text(TEXT_H, 12, W)
    big = Encoding(W, H, round(need), round(need * H / W))
    say(f"Width needed for 12 px text: {need:.0f} px; grid {big.grid}, {big.tokens:,} tokens, "
        f"{big.tokens / small.tokens:.1f}x the 448 px budget")
    x0, y0, x1, y1 = cell_box(GRID - 1, 0)
    crop = Encoding(x1 - x0, y1 - y0, 448, 336)
    say(f"Top-right crop {x1 - x0} x {y1 - y0} resized to 448 x 336: {crop.tokens} tokens, "
        f"text height {crop.text_height(TEXT_H):.2f} px")
    q448 = Encoding(W, H, 448, 336, merge=2)
    say(f"Qwen2-VL merges 2 x 2 patches: the same 448 x 336 image is {q448.tokens} language-model tokens")
    say("")
    say("Context arithmetic from the drift and compute-axes sections:")
    say(f"  768 visual + 5,000 reasoning tokens: image share {image_share(768, 5000):.1%}")
    for t in (500, 5000, 50000):
        say(f"  attention scores for T={t:,} after N_v=768: T*N_v + T(T+1)/2 = {attention_cost(768, t):,}")
    say(f"  a 768-token crop added at step 1000 of a 5,000-step trace is attended {4000:,} more times")

    # Measured on a rendered scene: digit height after the resize, and a strip of what the encoder gets.
    s = make_scene(0)
    px0, py0, px1, py1 = s.plaque_box
    say("")
    say("Rendered scene 0: the plaque region as each encoder width receives it (results/01_plaque_strip.png)")
    say(f"{'encoder width':>14} {'text px':>8} {'patches':>8} {'Qwen2-VL tokens':>16} {'detail kept':>12}")
    tiles = []
    for w in (448, 672, 896, 1344, 1512, 2016, 4032):
        h = round(w * H / W)
        e = Encoding(W, H, w, h)
        kept = detail_retained(s, View((0, 0, W, H), w, h))
        say(f"{w:>14} {e.text_height(TEXT_H):>8.1f} {e.patches:>8,} {Encoding(W, H, w, h, merge=2).tokens:>16,} "
            f"{kept:>12.2f}")
        small_img = resize_to(s.image, w, h)
        f = w / W
        region = small_img.crop((int(px0 * f), int(py0 * f), int(px1 * f) + 1, int(py1 * f) + 1))
        tiles.append((f"{w} px", region.resize((360, 60), Image.NEAREST)))
    c = resize_to(s.image.crop(cell_box(GRID - 1, 0)), 448, 336)
    f = 448 / (x1 - x0)
    region = c.crop((int((px0 - x0) * f), int((py0 - y0) * f), int((px1 - x0) * f) + 1, int((py1 - y0) * f) + 1))
    tiles.append(("crop @448", region.resize((360, 60), Image.NEAREST)))
    kept = detail_retained(s, View(cell_box(GRID - 1, 0), 448, 336))
    say(f"{'crop @448':>14} {crop.text_height(TEXT_H):>8.1f} {crop.patches:>8,} {192:>16,} {kept:>12.2f}")
    say("detail kept: correlation of horizontal intensity gradients between the original plaque and the view")
    say("scaled back up (1 = every letter edge kept, 0 = none). Widths here are not snapped to multiples of 28.")
    strip = Image.new("RGB", (480, 70 * len(tiles)), (255, 255, 255))
    d = ImageDraw.Draw(strip)
    for i, (name, t) in enumerate(tiles):
        d.text((4, 70 * i + 25), name, fill=(0, 0, 0))
        strip.paste(t, (115, 70 * i + 5))
    strip.save(LAB / "results" / "01_plaque_strip.png")
    s.image.resize((1008, 756), Image.BOX).save(LAB / "results" / "01_scene0.png")

    Path(args.out + ".txt").write_text("\n".join(lines) + "\n")
    Path(args.out + ".json").write_text(json.dumps({
        "full_tokens": full.patches, "small_tokens": small.tokens, "small_text_px": small.text_height(TEXT_H),
        "width_for_12px": need, "big_tokens": big.tokens, "crop_tokens": crop.tokens,
        "crop_text_px": crop.text_height(TEXT_H)}, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
