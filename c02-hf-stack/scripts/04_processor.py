"""Section 7: what AutoProcessor adds for a VLM. Prints real batch keys and shapes.

Run: python c02-hf-stack/scripts/04_processor.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image, ImageDraw
from transformers import AutoConfig, AutoImageProcessor, AutoProcessor

from hfstack.collators import VisionChatCollator
from hfstack.utils import SMALL_VLM, be_polite

be_polite()


def show(batch) -> None:
    for key, value in batch.items():
        print(f"  {key:<22} {tuple(value.shape)!s:<26} {str(value.dtype).replace('torch.', '')}")


def card(width: int, height: int, text: str) -> Image.Image:
    """A synthetic image: the label is exactly known."""
    image = Image.new("RGB", (width, height), (240, 240, 235))
    ImageDraw.Draw(image).text((20, 20), text, fill=(20, 20, 20))
    return image


processor = AutoProcessor.from_pretrained(SMALL_VLM)
config = AutoConfig.from_pretrained(SMALL_VLM)
print(type(processor).__name__, "=", type(processor.tokenizer).__name__, "+", type(processor.image_processor).__name__)

image = card(640, 480, "red square")
messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What does it say?"}]}]
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
print("\nprompt text:", repr(prompt))

print("\n--- 640x480 image, default settings ---")
batch = processor(text=[prompt], images=[[image]], return_tensors="pt")
show(batch)
ids = batch["input_ids"][0]
image_tokens = (ids == config.image_token_id).sum().item()
tiles = batch["pixel_values"].shape[1]
print(f"  image tokens in input_ids: {image_tokens} = {tiles} tiles x {image_tokens // tiles} tokens per tile")
print("  first 8 tokens:", processor.tokenizer.convert_ids_to_tokens(ids[:8].tolist()))

v = config.vision_config
per_tile = (v.image_size // v.patch_size) ** 2 // config.scale_factor**2
print(
    f"  tokens per tile from config: ({v.image_size}/{v.patch_size})^2 / {config.scale_factor}^2 = {per_tile}"
)

print("\n--- same image, do_image_splitting=False ---")
batch = processor(text=[prompt], images=[[image]], do_image_splitting=False, return_tensors="pt")
show(batch)
print("  image tokens:", (batch["input_ids"] == config.image_token_id).sum().item())

print("\n--- batch of two different images: pixel_values pad on the tile axis ---")
batch = processor(
    text=[prompt, prompt], images=[[card(640, 480, "a")], [card(300, 300, "b")]], padding=True, return_tensors="pt"
)
show(batch)
print("  real tiles per image:", batch["pixel_attention_mask"].flatten(2).any(-1).sum(1).tolist())

print("\n--- the collator for training: labels only on the assistant reply ---")
features = [
    {
        "image": card(300, 300, "seven"),
        "messages": messages
        + [{"role": "assistant", "content": [{"type": "text", "text": "It says seven."}]}],
    },
    {
        "image": card(640, 480, "two"),
        "messages": messages + [{"role": "assistant", "content": [{"type": "text", "text": "Two."}]}],
    },
]
batch = VisionChatCollator(processor)(features)
show(batch)
for row, labels in enumerate(batch["labels"]):
    kept = labels[labels != -100]
    print(f"  row {row}: {labels.numel()} positions, {kept.numel()} labelled ->", repr(processor.tokenizer.decode(kept)))

print("\n--- a different VLM family has different keys: Qwen2-VL's image processor ---")
# Only the image processor: the full Qwen2-VL processor also builds a video processor, which needs torchvision.
qwen = AutoImageProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
batch = qwen(images=[image], return_tensors="pt")
show(batch)
t, h, w = batch["image_grid_thw"][0].tolist()
print(f"  image_grid_thw: {[t, h, w]} -> {t * h * w} patches; merge_size {qwen.merge_size}")
print(f"  image_pad tokens the text needs: {t * h * w} / {qwen.merge_size}^2 = {t * h * w // qwen.merge_size**2}")
