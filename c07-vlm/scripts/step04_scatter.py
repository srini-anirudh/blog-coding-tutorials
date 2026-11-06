"""Step 4: scatter image features into the embedding sequence, naive loop against masked_scatter."""

import _common  # noqa: F401
import random
import time

import torch

from tinyvlm import build_tinyvlm, describe_device
from tinyvlm.data import render
from tinyvlm.merge import merge_masked_scatter, merge_naive

model, processor = build_tinyvlm()
image = render(2, "red", "circle", random.Random(0))
prompt = processor.apply_chat_template(
    [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What do you see?"}]}], add_generation_prompt=True
)
batch = processor(prompt, [image])
ids = batch["input_ids"]
mask = ids == model.config.image_token_id
print("prompt string           ", repr(prompt))
print("input_ids               ", tuple(ids.shape), " placeholders:", int(mask.sum()))
first = int(mask[0].float().argmax())
print("placeholder slots       ", first, "to", first + int(mask.sum()) - 1)

with torch.no_grad():
    text_embeds = model.get_input_embeddings()(ids)
    features = model.get_image_features(batch["pixel_values"])
    merged = merge_masked_scatter(text_embeds, ids, features, model.config.image_token_id)
print("text_embeds             ", tuple(text_embeds.shape))
print("image features          ", tuple(features.shape))
print("merged                  ", tuple(merged.shape))
print("slot", first, "equals features[0, 0]   ", torch.equal(merged[0, first], features[0, 0]))
print("slot", first + 48, "equals features[0, 48]  ", torch.equal(merged[0, first + 48], features[0, 48]))
print("text rows untouched      ", torch.equal(merged[~mask], text_embeds[~mask]))

# A batch big enough for the difference to matter: 8 samples of 63 tokens, 49 of them placeholders.
big_ids, big_embeds, big_features = ids.repeat(8, 1), text_embeds.repeat(8, 1, 1), features.repeat(8, 1, 1)
for name, fn in (("merge_naive", merge_naive), ("merge_masked_scatter", merge_masked_scatter)):
    seconds = []
    for _ in range(20):
        start = time.perf_counter()
        out = fn(big_embeds, big_ids, big_features, model.config.image_token_id)
        seconds.append(time.perf_counter() - start)
    print(f"{name:<22} {min(seconds) * 1000:8.3f} ms per batch, best of 20 ({describe_device(big_ids.device)})")
