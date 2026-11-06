"""Load SmolVLM-256M-Instruct and turn (image, question) into model inputs.

SmolVLM is an Idefics3 model: a SigLIP vision encoder (86M), a pixel-shuffle
connector (7M) and a SmolLM2 decoder (135M plus a 28M untied LM head).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"


def load_processor(model_id: str = MODEL_ID):
    # One 512x512 tile instead of 16 tiles + a global view: 64 image tokens, not 1,088.
    # Our synthetic scenes are 512x512 with large objects, so tiling buys nothing.
    proc = AutoProcessor.from_pretrained(model_id)
    proc.image_processor.do_image_splitting = False
    return proc


def load_model(model_id: str = MODEL_ID, device: torch.device | str = "cpu", dtype=torch.float32):
    model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=dtype)
    return model.to(device)


def chat_prompt(processor, question: str) -> str:
    """'<|im_start|>User:<image>{question}<end_of_utterance>\\nAssistant:'"""
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
    return processor.apply_chat_template(msgs, add_generation_prompt=True)


@dataclass
class PromptInputs:
    input_ids: torch.Tensor       # (1, T_prompt) with 64 <image> placeholder ids inside
    pixel_values: torch.Tensor    # (1, 1, 3, 512, 512)

    def to(self, device) -> "PromptInputs":
        return PromptInputs(self.input_ids.to(device), self.pixel_values.to(device))


def encode_prompt(processor, image: Image.Image, question: str) -> PromptInputs:
    enc = processor(text=chat_prompt(processor, question), images=[image], return_tensors="pt")
    return PromptInputs(enc["input_ids"], enc["pixel_values"])


def encode_answer(processor, answer: str) -> torch.Tensor:
    """Answer token ids, ending with <end_of_utterance>. Shape (T_answer,).

    We tokenize the answer on its own and append it to the prompt ids. That keeps
    the prompt ids byte-identical to what generate() conditions on, so the
    log-probabilities we train on are the ones the model would sample from.
    """
    tok = processor.tokenizer
    ids = tok(" " + answer, add_special_tokens=False)["input_ids"]
    return torch.tensor(ids + [end_of_utterance_id(processor)], dtype=torch.long)


def end_of_utterance_id(processor) -> int:
    return processor.tokenizer.convert_tokens_to_ids("<end_of_utterance>")


@torch.no_grad()
def generate(model, processor, image: Image.Image, question: str, max_new_tokens: int = 12, **kw) -> list[str]:
    """Greedy (default) or sampled answers for one image. Returns decoded strings."""
    p = encode_prompt(processor, image, question).to(model.device)
    out = model.generate(input_ids=p.input_ids, attention_mask=torch.ones_like(p.input_ids),
                         pixel_values=p.pixel_values, max_new_tokens=max_new_tokens,
                         eos_token_id=end_of_utterance_id(processor),
                         pad_token_id=processor.tokenizer.pad_token_id, **kw)
    return processor.batch_decode(out[:, p.input_ids.shape[1]:], skip_special_tokens=True)
