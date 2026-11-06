"""Break it on purpose: the failures that raise, and the ones that do not.

A processor built for 196 tokens per crop meets a model that makes 49. Then we feed labels that supervise
every token, and compare the loss with the correct labels on an untrained projector. Last, a 12 MP photo
with 48 tiles and no pixel shuffle: nothing raises, even past the language model's position limit. It is just slow.
"""

import _common  # noqa: F401
import random
import time

import torch
from PIL import Image

from tinyvlm import TinyVLMProcessor, build_tinyvlm, describe_device
from tinyvlm.data import render

model, processor = build_tinyvlm()
image = render(2, "red", "circle", random.Random(0))
messages = [
    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What do you see?"}]},
    {"role": "assistant", "content": "Two red circles."},
]

print("1. Processor and model disagree about tokens per crop")
wrong = TinyVLMProcessor(processor.image_processor, processor.tokenizer, image_seq_len=196)
batch = wrong.collate([wrong.build_example(messages, [image])])
try:
    model(batch["input_ids"], batch["pixel_values"], batch["attention_mask"])
except ValueError as error:
    print("   ValueError:", error)

print("2. The marker is missing from the text")
try:
    processor("<|im_start|>user\nWhat do you see?<|im_end|>\n", [image])
except ValueError as error:
    print("   ValueError:", error)

print("3. Wrong loss mask (fresh projector, one example)")
batch = processor.collate([processor.build_example(messages, [image])])
with torch.no_grad():
    good = model(batch["input_ids"], batch["pixel_values"], batch["attention_mask"], labels=batch["labels"]).loss
    everything = batch["input_ids"].clone()
    bad = model(batch["input_ids"], batch["pixel_values"], batch["attention_mask"], labels=everything).loss
print(f"   labels on the reply only: {good:.3f} over {(batch['labels'] != -100).sum().item()} targets")
print(f"   labels on every token:    {bad:.3f} over {everything.numel()} targets, {(everything == processor.image_token_id).sum().item()} of them <image>")

print("4. Excessive image resolution (one 4032 x 3024 photo)")
photo = Image.new("RGB", (4032, 3024), "white")
prompt = processor.apply_chat_template(messages[:1], add_generation_prompt=True)
big_model, big_processor = build_tinyvlm(pixel_shuffle_factor=1, strategy="tile", max_tiles=48)
limit = model.config.text_config.max_position_embeddings
print(f"   language model max_position_embeddings: {limit} | {describe_device(batch['input_ids'].device)}")
for name, m, p in (("resize, s = 2", model, processor), ("48 tiles, s = 1", big_model, big_processor)):
    batch = p(prompt, [photo])
    start = time.perf_counter()
    with torch.no_grad():
        out = m(batch["input_ids"], batch["pixel_values"], batch["attention_mask"], use_cache=True)
    seconds = time.perf_counter() - start
    cache_mb = sum(c.keys.numel() + c.values.numel() for c in out.past_key_values.layers) * 4 / 1e6
    print(f"   {name:<16} {batch['pixel_values'].shape[0]:>3} crops  {batch['input_ids'].shape[1]:>5} tokens  "
          f"{seconds:7.2f} s  KV cache {cache_mb:7.1f} MB  past the limit: {batch['input_ids'].shape[1] > limit}")
