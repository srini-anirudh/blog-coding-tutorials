"""Break it on purpose: the failures that raise, and one that does not.

A processor built for 196 tokens per crop meets a model that makes 49. Then we feed labels that supervise
every token, and compare the loss with the correct labels on an untrained projector.
"""

import _common  # noqa: F401
import random

import torch

from tinyvlm import TinyVLMProcessor, build_tinyvlm
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
