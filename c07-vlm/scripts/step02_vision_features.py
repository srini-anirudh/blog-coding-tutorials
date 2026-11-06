"""Step 2: what the pretrained encoder returns for a batch of pictures."""

import _common  # noqa: F401
import random

import torch
from transformers import SiglipVisionConfig, SiglipVisionModel

from _common import stats
from tinyvlm import VISION_ID
from tinyvlm.data import render
from tinyvlm.image_processing import TinyVLMImageProcessor

config = SiglipVisionConfig.from_pretrained(VISION_ID)
config.vision_use_head = False  # keep every patch token; do not pool them into one vector
vision = SiglipVisionModel.from_pretrained(VISION_ID, config=config).eval()
print(f"encoder parameters      {sum(p.numel() for p in vision.parameters()):,}")
print(f"image {config.image_size}, patch {config.patch_size} -> {(config.image_size // config.patch_size) ** 2} tokens, d_v = {config.hidden_size}")

images = [render(1, "red", "circle", random.Random(0)), render(3, "blue", "square", random.Random(1))]
pixels = TinyVLMImageProcessor()(images)["pixel_values"]
print(stats("pixel_values", pixels))
with torch.no_grad():
    out = vision(pixel_values=pixels, output_hidden_states=True)
print(stats("last_hidden_state", out.last_hidden_state))
print(stats("hidden_states[-2]", out.hidden_states[-2]))
print("hidden_states           ", len(out.hidden_states), "tensors (patch embeddings + 12 layers)")

# Two patch tokens of the same picture: background against background, background against shape.
tokens = torch.nn.functional.normalize(out.last_hidden_state[0], dim=-1)
sim = tokens @ tokens.T  # 196 x 196 cosine similarity
print(f"cosine(corner, other corner)  {sim[0, 13]:+.3f}")
print(f"mean cosine of all pairs      {sim.mean():+.3f}")
