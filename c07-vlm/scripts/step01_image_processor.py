"""Step 1: our image processor against Hugging Face's SigLIP processor, on a real rendered picture."""

import _common  # noqa: F401
import random

from transformers.models.siglip.image_processing_pil_siglip import SiglipImageProcessorPil

from _common import stats
from tinyvlm.data import render
from tinyvlm.image_processing import TinyVLMImageProcessor

image = render(2, "red", "circle", random.Random(0), size=480).crop((0, 60, 480, 420))  # 480 x 360, not square
print("input picture          ", image.size, image.mode)

ours = TinyVLMImageProcessor()([image])
hf = SiglipImageProcessorPil(size={"height": 224, "width": 224}, resample=3)(image, return_tensors="pt")

print(stats("ours pixel_values", ours["pixel_values"]))
print(stats("hf   pixel_values", hf["pixel_values"]))
print(f"max |ours - hf|         {(ours['pixel_values'] - hf['pixel_values']).abs().max():.2e}")
print("first pixel, ours       ", ours["pixel_values"][0, :, 0, 0].tolist())
