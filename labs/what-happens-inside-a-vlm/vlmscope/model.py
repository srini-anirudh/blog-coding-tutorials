"""A thin wrapper around a small Hugging Face VLM that exposes what the experiments need.

The default model is SmolVLM-256M-Instruct: a SigLIP encoder (12 layers, 512 px, 16 px patches), a pixel-shuffle
connector that folds each 4 x 4 block of patches into one token, and SmolLM2-135M (30 layers). With image splitting
off, one 512 x 512 image becomes 64 tokens on an 8 x 8 grid in raster order, placed before the question.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import torch
from PIL import Image

from . import intervene as iv_mod
from .intervene import Intervention, intervene

MODEL_ID = os.environ.get("VLMSCOPE_MODEL", "HuggingFaceTB/SmolVLM-256M-Instruct")
BASE_LM_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
SUFFIX = " Answer with one word."


@dataclass
class Prompt:
    input_ids: torch.Tensor          # [1, seq]
    extra: dict                      # pixel_values etc.
    image_pos: torch.Tensor          # positions of image tokens, raster order
    question_pos: torch.Tensor       # text after the image, excluding the last position
    last: int                        # the position that predicts the answer
    tokens: list[str]
    embeds: torch.Tensor | None = None   # [1, seq, d] input embeddings, image features already scattered in

    def find(self, word: str) -> int:
        """Position of the first question token whose text contains `word` (e.g. 'circle')."""
        for p in self.question_pos.tolist():
            if word.lower() in self.tokens[p].lower():
                return p
        raise KeyError(word)


class VLM:
    def __init__(self, model_id: str = MODEL_ID, device: str | None = None, dtype=torch.float32):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, dtype=dtype, attn_implementation=iv_mod.NAME).to(device).eval()
        self.device = device
        self.tok = self.processor.tokenizer
        self.image_token_id = self.model.config.image_token_id
        inner = self.model.model
        self.text = inner.text_model
        self.vision = inner.vision_model
        self.connector = inner.connector
        self.layers = self.text.layers
        self.norm = self.text.norm
        self.lm_head = self.model.lm_head
        self.n_layers = len(self.layers)
        self.grid = 8
        self._handles = iv_mod.install(self.text)

    # ---- prompts -------------------------------------------------------------------------------------------
    def prompt(self, image: Image.Image | None, question: str, suffix: str = SUFFIX) -> Prompt:
        content = ([{"type": "image"}] if image is not None else []) + [{"type": "text", "text": question + suffix}]
        text = self.processor.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True)
        if image is not None:
            enc = self.processor(text=text, images=[image], return_tensors="pt", do_image_splitting=False)
        else:
            enc = self.processor(text=text, return_tensors="pt")
        ids = enc.pop("input_ids").to(self.device)
        enc.pop("attention_mask", None)
        extra = {k: v.to(self.device) for k, v in enc.items()}
        img = (ids[0] == self.image_token_id).nonzero().flatten()
        start = int(img[-1]) + 2 if len(img) else 4   # skip the closing <fake_token_around_image>
        last = ids.shape[1] - 1
        p = Prompt(ids, extra, img, torch.arange(start, last, device=self.device), last,
                   self.tok.convert_ids_to_tokens(ids[0]))
        # Image tokens never see the question (causal mask), and the encoder does not either, so the vision
        # tower runs once per prompt. Every later intervention on this prompt only re-runs the language model.
        p.embeds = self.embeddings(p)[None]
        return p

    def extend(self, p: Prompt, token_id: int) -> Prompt:
        """The same prompt with one token appended (e.g. a space, so the next token is a bare digit)."""
        ids = torch.cat([p.input_ids, torch.tensor([[token_id]], device=self.device)], 1)
        emb = self.text.get_input_embeddings()(ids[:, -1:])
        return Prompt(ids, p.extra, p.image_pos, torch.cat([p.question_pos, torch.tensor([p.last], device=self.device)]),
                      p.last + 1, p.tokens + [self.tok.convert_ids_to_tokens(token_id)], torch.cat([p.embeds, emb], 1))

    def word_ids(self, words: list[str]) -> torch.Tensor:
        """First token id of ' Word' (the model answers with a leading space and a capital)."""
        ids = [self.tok(" " + w.capitalize(), add_special_tokens=False).input_ids[0] for w in words]
        assert len(set(ids)) == len(ids), f"answer words share a first token: {words}"
        return torch.tensor(ids, device=self.device)

    # ---- forward passes ------------------------------------------------------------------------------------
    @torch.no_grad()
    def run(self, p: Prompt, iv: Intervention | None = None) -> torch.Tensor:
        """Logits at the last position, [vocab]. The intervention (if any) is active during the pass."""
        iv = iv or Intervention()
        with intervene(iv):
            if p.embeds is not None:
                out = self.model(inputs_embeds=p.embeds, use_cache=False)
            else:
                out = self.model(input_ids=p.input_ids, **p.extra, use_cache=False)
        return out.logits[0, -1].float()

    @torch.no_grad()
    def embeddings(self, p: Prompt) -> torch.Tensor:
        """The input embeddings the language model sees, image features already scattered in: [seq, d]."""
        iv = Intervention(capture_residual=True)
        with intervene(iv):
            self.model(input_ids=p.input_ids, **p.extra, use_cache=False)
        return iv.residual[-1][0]

    @torch.no_grad()
    def vision_features(self, image: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        """(encoder patch features [1024, 768], connector output [64, d_lm]) for one image."""
        enc = self.processor.image_processor(images=[image], return_tensors="pt", do_image_splitting=False)
        pix = enc["pixel_values"].to(self.device)[0]           # [1, 3, 512, 512]
        patches = self.vision(pixel_values=pix).last_hidden_state
        return patches[0], self.connector(patches)[0]

    def lens(self, h: torch.Tensor) -> torch.Tensor:
        """Logit lens: W_U . LN(h) for any residual vector(s)."""
        return self.lm_head(self.norm(h)).float()
