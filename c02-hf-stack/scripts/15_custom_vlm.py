"""Section 23: what a custom VLM adds: two sub-configs, an image token, a scatter in forward.

Run: python c02-hf-stack/scripts/15_custom_vlm.py
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import AutoModelForImageTextToText

from hfstack import custom_model, custom_vlm
from hfstack.custom_vlm import BackwardPassVLMConfig, BackwardPassVLMForConditionalGeneration, expand_image_tokens
from hfstack.utils import be_polite, count_parameters

be_polite()
torch.manual_seed(0)
custom_model.register()
custom_vlm.register()

config = BackwardPassVLMConfig(
    vision_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2, image_size=32, patch_size=8),
    text_config=dict(vocab_size=512),
)
model = BackwardPassVLMForConditionalGeneration(config).eval()
print("vision_config:", type(config.vision_config).__name__, "| text_config:", type(config.text_config).__name__)
print("parameters:", f"{count_parameters(model)[1]:,}")

patches = (config.vision_config.image_size // config.vision_config.patch_size) ** 2
ids = [1, config.image_token_id, 50, 60, 70]
expanded = expand_image_tokens(ids, config.image_token_id, patches)
print(f"\nprocessor step: {len(ids)} tokens with 1 placeholder -> {len(expanded)} tokens ({patches} image tokens)")

input_ids = torch.tensor([expanded])
pixels = torch.randn(1, 3, 32, 32)
labels = input_ids.clone()
labels[input_ids == config.image_token_id] = -100
with torch.no_grad():
    features = model.image_features(pixels)
    out = model(input_ids=input_ids, pixel_values=pixels, labels=labels)
print("pixel_values  ", tuple(pixels.shape))
print("image features", tuple(features.shape))
print("input_ids     ", tuple(input_ids.shape))
print("logits        ", tuple(out.logits.shape), "| loss", round(out.loss.item(), 4))

with tempfile.TemporaryDirectory() as tmp:
    model.save_pretrained(tmp)
    saved = json.loads((Path(tmp) / "config.json").read_text())
    print("\nconfig.json keys:", sorted(k for k in saved if k != "transformers_version"))
    print("vision_config.model_type:", saved["vision_config"]["model_type"])
    print("text_config.model_type  :", saved["text_config"]["model_type"])
    loaded = AutoModelForImageTextToText.from_pretrained(tmp).eval()
    with torch.no_grad():
        again = loaded(input_ids=input_ids, pixel_values=pixels).logits
    print("Auto class ->", type(loaded).__name__, "| round trip logits match:", torch.allclose(again, out.logits))

tokens = model.generate(input_ids=input_ids, pixel_values=pixels, max_new_tokens=5, do_sample=False)
print("\ngenerate:", tuple(input_ids.shape), "->", tuple(tokens.shape))

print("\n--- break it: the text has fewer image tokens than the encoder produced ---")
try:
    model(input_ids=torch.tensor([ids]), pixel_values=pixels)
except ValueError as err:
    print("ValueError:", err)
