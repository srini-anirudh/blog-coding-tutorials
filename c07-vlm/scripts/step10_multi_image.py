"""Step 10: two pictures in one prompt, and how the processor records which token belongs to which."""

import _common  # noqa: F401
import random

import torch

from tinyvlm import build_tinyvlm
from tinyvlm.data import render

model, processor = build_tinyvlm()
images = [render(1, "red", "circle", random.Random(0)), render(3, "blue", "square", random.Random(1))]
messages = [{"role": "user", "content": [
    {"type": "text", "text": "Image 1: "}, {"type": "image"},
    {"type": "text", "text": "\nImage 2: "}, {"type": "image"},
    {"type": "text", "text": "\nWhich one has more shapes?"},
]}]
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
batch = processor(prompt, images)
index = batch["image_index"][0]
print("input_ids", tuple(batch["input_ids"].shape), "| pixel_values", tuple(batch["pixel_values"].shape), "| num_crops", batch["num_crops"])
for k in range(2):
    where = (index == k).nonzero().flatten()
    print(f"image {k}: tokens {where[0].item()} to {where[-1].item()} ({where.numel()} tokens)")
print("text between images:", repr(processor.tokenizer.decode(batch["input_ids"][0][(index == 0).nonzero().max() + 1 : (index == 1).nonzero().min()])))
with torch.no_grad():
    features = model.get_image_features(batch["pixel_values"])
    embeds = model.embed(batch["input_ids"], batch["pixel_values"])
print("features", tuple(features.shape), "| embeds", tuple(embeds.shape))
print("image 1 slot holds features[1, 0]:", torch.equal(embeds[0, (index == 1).nonzero()[0].item()], features[1, 0]))
