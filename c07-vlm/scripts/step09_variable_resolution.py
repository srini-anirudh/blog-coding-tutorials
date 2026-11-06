"""Step 9: how each resolution strategy turns picture sizes into image tokens. Real crops, no model."""

import _common  # noqa: F401
from PIL import Image

from tinyvlm.image_processing import TinyVLMImageProcessor, select_grid

SIZES = [(224, 224), (640, 480), (1000, 300), (300, 1000), (1344, 896), (2048, 1536), (100, 50)]
print(f"{'picture':>11} | {'resize':>7} | {'pad':>7} | {'tile grid':>9} {'crops':>5} {'s=1':>6} {'s=2':>6}")
for w, h in SIZES:
    image = Image.new("RGB", (w, h), "white")
    n_resize = TinyVLMImageProcessor(strategy="resize")([image])["pixel_values"].shape[0]
    n_pad = TinyVLMImageProcessor(strategy="pad")([image])["pixel_values"].shape[0]
    tiled = TinyVLMImageProcessor(strategy="tile", max_tiles=6)([image])["pixel_values"].shape[0]
    cols, rows = select_grid(w, h, 224, 6)
    print(f"{w:>5} x {h:<4} | {n_resize * 49:>7} | {n_pad * 49:>7} | {cols:>4} x {rows:<2} {tiled:>5} {tiled * 196:>6} {tiled * 49:>6}")
print("resize and pad columns use pixel shuffle s = 2 (49 tokens per crop); s = 1 gives 196.")

# What each strategy does to a wide picture: a fixed resize distorts it, padding shrinks it.
wide = Image.new("RGB", (1000, 300), "white")
print("resize  ", TinyVLMImageProcessor(strategy="resize").crops(wide)[0].size, "aspect ratio changes by", round((1000 / 300) / 1, 2), "x")
print("pad     ", TinyVLMImageProcessor(strategy="pad").crops(wide)[0].size, "content fills", round(300 / 1000 * 100), "% of the crop height")
