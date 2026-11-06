"""Step 12: save_pretrained, reload through AutoModel, and compare logits."""

import _common  # noqa: F401
import os
import random
import sys
import tempfile

import torch
from transformers import AutoConfig, AutoModel

from tinyvlm import TinyVLMProcessor, build_tinyvlm
from tinyvlm.data import render

model, processor = build_tinyvlm()
with tempfile.TemporaryDirectory() as directory:
    model.save_pretrained(directory)
    processor.save_pretrained(directory)
    for name in sorted(os.listdir(directory)):
        print(f"  {name:<28} {os.path.getsize(os.path.join(directory, name)) / 1e6:9.2f} MB")
    config = AutoConfig.from_pretrained(directory)
    print("AutoConfig ->", type(config).__name__, "| image_token_id", config.image_token_id, "| image_seq_len", config.image_seq_len)
    again = AutoModel.from_pretrained(directory).eval()
    processor2 = TinyVLMProcessor.from_pretrained(directory)
    print("AutoModel  ->", type(again).__name__, "| tied:", again.lm.lm_head.weight.data_ptr() == again.lm.model.embed_tokens.weight.data_ptr())
    prompt = processor2.apply_chat_template(
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What do you see?"}]}], add_generation_prompt=True
    )
    batch = processor2(prompt, [render(2, "red", "circle", random.Random(0))])
    with torch.no_grad():
        a = model(batch["input_ids"], batch["pixel_values"], batch["attention_mask"]).logits
        b = again(batch["input_ids"], batch["pixel_values"], batch["attention_mask"]).logits
    print("logits identical:", torch.equal(a, b), tuple(a.shape))
sys.stdout.flush()
