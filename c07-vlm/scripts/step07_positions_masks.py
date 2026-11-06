"""Step 7 and 8: position ids and the attention mask, on a short batch we can read."""

import _common  # noqa: F401
from PIL import Image

from tinyvlm.masks import build_attention_mask, render
from tinyvlm.positions import position_ids_1d, position_ids_3d
from tinyvlm.testing import tiny_vlm

_, processor = tiny_vlm(pixel_shuffle_factor=2)  # tiny model: 4 image tokens per crop, one token per character
prompts = ["<image>hi", "<image>a"]
batch = processor(prompts, [[Image.new("RGB", (40, 30), "red")], [Image.new("RGB", (40, 30), "blue")]], padding_side="left")
print("input_ids\n", batch["input_ids"])
print("attention_mask\n", batch["attention_mask"])
print("image_index\n", batch["image_index"])
print("position_ids (1D)\n", position_ids_1d(batch["attention_mask"]))

labels = ["pad" if m == 0 else ("img" if k >= 0 else "txt") for m, k in zip(batch["attention_mask"][1].tolist(), batch["image_index"][1].tolist())]
print("\nsample 1, causal + padding (rows are queries, columns are keys)")
print(render(build_attention_mask(batch["attention_mask"], batch["image_index"])[1, 0], labels))
print("\nsample 1, with the image block bidirectional")
print(render(build_attention_mask(batch["attention_mask"], batch["image_index"], bidirectional_images=True)[1, 0], labels))

image_index = batch["image_index"][0]
print("\n3D positions for sample 0 (t, h, w), image grid 2 x 2")
print(position_ids_3d(image_index, grid=(2, 2)))
