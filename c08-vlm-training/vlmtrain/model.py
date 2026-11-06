"""A minimal VLM that follows the C7 `tinyvlm` contract, so this folder runs on its own.

    TinyVLM
      .vision     SigLIP vision tower (pretrained google/siglip-base-patch16-224, 86M parameters)
      .projector  pixel shuffle 2x2 + MLP, d_v * 4 -> d_lm (new, about 2M parameters)
      .lm         SmolLM2-135M-Instruct, a Llama-architecture decoder

    forward(input_ids, pixel_values, attention_mask=None, labels=None) -> .loss, .logits

C7 builds this model step by step, and the trainer uses C7's package whenever `../c07-vlm` exists. This
file is the fallback that lets the folder run alone. tests/test_real_weights.py checks that the two
compute the same logits from the same weights.

Shapes for one 224 px tile:
    pixel_values      (n_tiles, 3, 224, 224)
    vision output     (n_tiles, 196, 768)     14 x 14 patches of 16 px
    pixel shuffle     (n_tiles, 49, 3072)     each 2 x 2 patch block folded into channels
    projector output  (n_tiles, 49, 576)      one LM embedding per visual token
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM, SiglipVisionConfig, SiglipVisionModel


@dataclass
class VLMConfig:
    vision_name: str = "google/siglip-base-patch16-224"
    lm_name: str = "HuggingFaceTB/SmolLM2-135M-Instruct"
    vision_config: dict | None = None  # set both configs to build random towers (offline tests)
    lm_config: dict | None = None
    image_size: int = 224
    patch_size: int = 16
    pixel_shuffle: int = 2
    projector: str = "mlp"  # "linear" | "mlp"
    image_token_id: int = -1  # filled in from the tokenizer by build_vlm
    extra: dict = field(default_factory=dict)

    @property
    def tokens_per_tile(self) -> int:
        grid = self.image_size // self.patch_size
        return (grid // self.pixel_shuffle) ** 2


def pixel_shuffle(x: torch.Tensor, s: int) -> torch.Tensor:
    """(N, g*g, d) -> (N, (g/s)^2, d*s*s): fold each s x s block of neighbouring patches into channels.

    Token count drops by s^2 and no information is thrown away. This is SmolVLM's and Idefics3's
    token-compression step.
    """
    n, t, d = x.shape
    g = int(t**0.5)
    x = x.view(n, g // s, s, g // s, s, d)          # (N, rows, s, cols, s, d)
    x = x.permute(0, 1, 3, 2, 4, 5)                 # (N, rows, cols, s, s, d)
    return x.reshape(n, (g // s) ** 2, s * s * d)


class Projector(nn.Module):
    def __init__(self, d_vision: int, d_lm: int, shuffle: int = 2, kind: str = "mlp"):
        super().__init__()
        self.shuffle = shuffle
        d_in = d_vision * shuffle * shuffle
        if kind == "linear":
            self.net = nn.Sequential(nn.Linear(d_in, d_lm))
        else:
            self.net = nn.Sequential(nn.Linear(d_in, d_lm), nn.GELU(), nn.Linear(d_lm, d_lm))

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """(N, T, d_v) -> (N, T / s^2, d_lm)"""
        return self.net(pixel_shuffle(feats, self.shuffle))


class TinyVLM(nn.Module):
    def __init__(self, config: VLMConfig):
        super().__init__()
        self.config = config
        if config.vision_config is not None:
            self.vision = SiglipVisionModel(SiglipVisionConfig(**config.vision_config, vision_use_head=False))
        else:
            # The attention-pooling head is only used for CLIP-style retrieval. A VLM reads every patch.
            vision_config = SiglipVisionConfig.from_pretrained(config.vision_name)
            vision_config.vision_use_head = False
            self.vision = SiglipVisionModel.from_pretrained(config.vision_name, config=vision_config, dtype=torch.float32)
        if config.lm_config is not None:
            self.lm = LlamaForCausalLM(LlamaConfig(**config.lm_config))
        else:
            self.lm = AutoModelForCausalLM.from_pretrained(config.lm_name, dtype=torch.float32)  # SmolLM2 ships as bf16
        self.projector = Projector(self.vision.config.hidden_size, self.lm.config.hidden_size,
                                   config.pixel_shuffle, config.projector)

    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """(n_tiles, 3, H, W) -> (n_tiles, tokens_per_tile, d_lm)"""
        return self.projector(self.vision(pixel_values=pixel_values).last_hidden_state)

    def embed(self, input_ids: torch.Tensor, pixel_values: torch.Tensor | None) -> torch.Tensor:
        """Token embeddings with every <image> position replaced by a projected visual token, in order."""
        emb = self.lm.get_input_embeddings()(input_ids)                         # (B, N, d_lm)
        if pixel_values is None:
            return emb
        feats = self.encode_images(pixel_values.to(emb.dtype))                  # (n_tiles, T, d_lm)
        slots = input_ids == self.config.image_token_id                         # (B, N)
        n_slots, n_feats = int(slots.sum()), feats.shape[0] * feats.shape[1]
        if n_slots != n_feats:
            raise ValueError(f"{n_slots} <image> tokens in the text but {n_feats} visual tokens from the "
                             f"images: the processor and the model disagree on tokens per tile")
        return emb.masked_scatter(slots.unsqueeze(-1), feats.reshape(-1, feats.shape[-1]).to(emb.dtype))

    def forward(self, input_ids: torch.Tensor, pixel_values: torch.Tensor | None = None,
                attention_mask: torch.Tensor | None = None, labels: torch.Tensor | None = None,
                position_ids: torch.Tensor | None = None):
        emb = self.embed(input_ids, pixel_values)
        return self.lm(inputs_embeds=emb, attention_mask=attention_mask, position_ids=position_ids, labels=labels)

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, pixel_values: torch.Tensor | None = None,
                 attention_mask: torch.Tensor | None = None, **generate_kwargs) -> torch.Tensor:
        """Greedy decoding from the LM's own generate(), fed with multimodal embeddings.

        Prompts must be left-padded. With inputs_embeds, HF returns only the new tokens.
        """
        emb = self.embed(input_ids, pixel_values)
        return self.lm.generate(inputs_embeds=emb, attention_mask=attention_mask, **generate_kwargs)


def build_vlm(config: VLMConfig, tokenizer) -> TinyVLM:
    """Build the model and grow the LM vocabulary for <image> and the location tokens.

    New embedding rows start at the mean of the old rows (HF's mean_resizing), so new tokens begin
    as "average" tokens instead of random vectors with a much larger norm than the trained ones.
    """
    config.image_token_id = tokenizer.convert_tokens_to_ids("<image>")
    model = TinyVLM(config)
    if model.lm.get_input_embeddings().num_embeddings != len(tokenizer):
        model.lm.resize_token_embeddings(len(tokenizer), mean_resizing=True)
    return model


def tiny_config() -> VLMConfig:
    """A random 2-layer SigLIP + 2-layer Llama: the same code paths in a model small enough for CI."""
    return VLMConfig(
        vision_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
                           image_size=32, patch_size=8),
        lm_config=dict(vocab_size=320, hidden_size=48, intermediate_size=96, num_hidden_layers=2,
                       num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=1024,
                       tie_word_embeddings=True),
        image_size=32, patch_size=8,
    )


def save_trainable(model: nn.Module, path: str | Path) -> int:
    """Save only parameters that train (a Stage 1 checkpoint is just the 2M projector weights)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {n: p.detach().cpu().contiguous() for n, p in model.named_parameters() if p.requires_grad}
    save_file(state, str(path))
    return sum(t.numel() for t in state.values())


def load_partial(model: nn.Module, path: str | Path) -> list[str]:
    """Load whatever tensors the file has (strict=False) and return their names."""
    state = load_file(str(path))
    missing_from_model = [k for k in state if k not in dict(model.named_parameters())]
    if missing_from_model:
        raise KeyError(f"checkpoint has unknown tensors, e.g. {missing_from_model[:3]}")
    model.load_state_dict(state, strict=False)
    return list(state)
