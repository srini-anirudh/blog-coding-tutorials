"""The extra pieces a custom vision-language model needs, on top of the custom LM.

    vision config + text config   two sub-configs nested inside one config.json
    image token ID                a placeholder token; one copy per visual feature vector
    processor step                expand one placeholder into N image tokens (N = patches per image)
    multimodal forward            embed text, encode image, scatter image features into the placeholders

C7 builds a full VLM and its processor from scratch. This file shows the minimum that makes one
integrate with the Auto classes and save/load like any other model.
"""

# No `from __future__ import annotations`: transformers reads the `config:` annotation at class creation.
import torch
from torch import nn
from transformers import (
    AutoConfig,
    AutoModelForImageTextToText,
    GenerationMixin,
    PreTrainedConfig,
    PreTrainedModel,
    SiglipVisionConfig,
    SiglipVisionModel,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

from .custom_model import BackwardPassConfig, BackwardPassForCausalLM


class BackwardPassVLMConfig(PreTrainedConfig):
    model_type = "backward_pass_vlm"
    # Sub-configs: each is saved as a nested dict in config.json and rebuilt as its own config class.
    sub_configs_defaults = {"vision_config": SiglipVisionConfig, "text_config": BackwardPassConfig}

    vision_config: dict | PreTrainedConfig | None = None
    text_config: dict | PreTrainedConfig | None = None
    image_token_id: int = 3  # a token the tokenizer reserves for "an image feature goes here"


def expand_image_tokens(input_ids: list[int], image_token_id: int, tokens_per_image: int) -> list[int]:
    """The processor's core job: replace each single image placeholder with `tokens_per_image` copies."""
    out: list[int] = []
    for tok in input_ids:
        out.extend([tok] * tokens_per_image if tok == image_token_id else [tok])
    return out


class BackwardPassVLMForConditionalGeneration(PreTrainedModel, GenerationMixin):
    config: BackwardPassVLMConfig
    base_model_prefix = "model"
    _no_split_modules = ["BackwardPassBlock"]
    _tied_weights_keys = {"language_model.lm_head.weight": "language_model.model.embed_tokens.weight"}

    def __init__(self, config: BackwardPassVLMConfig):
        super().__init__(config)
        self.vision = SiglipVisionModel(config.vision_config)
        v, d = config.vision_config.hidden_size, config.text_config.hidden_size
        self.projector = nn.Sequential(nn.Linear(v, d), nn.GELU(), nn.Linear(d, d))
        self.language_model = BackwardPassForCausalLM(config.text_config)
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.language_model.get_input_embeddings()

    def image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """pixel_values: (B_img, 3, H, W) → (B_img, P, D_text), P = (H / patch)²."""
        patches = self.vision(pixel_values=pixel_values).last_hidden_state  # (B_img, P, D_vision)
        return self.projector(patches)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.LongTensor | None = None,
        past_key_values=None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool | None = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)  # (B, T, D)
        if pixel_values is not None:
            features = self.image_features(pixel_values)  # (B_img, P, D)
            slots = input_ids == self.config.image_token_id  # (B, T)
            if slots.sum() != features.shape[0] * features.shape[1]:
                raise ValueError(
                    f"{int(slots.sum())} image tokens in the text but {features.shape[0] * features.shape[1]} "
                    "image features: the processor and the vision encoder disagree on tokens per image"
                )
            # Fill the placeholder positions, in order, with the image feature vectors.
            inputs_embeds = inputs_embeds.masked_scatter(slots[..., None], features.to(inputs_embeds.dtype))
        return self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )

    def prepare_inputs_for_generation(self, *args, pixel_values=None, is_first_iteration=False, **kwargs):
        inputs = super().prepare_inputs_for_generation(*args, is_first_iteration=is_first_iteration, **kwargs)
        if is_first_iteration:  # images are encoded once, during prefill; decode steps reuse the KV cache
            inputs["pixel_values"] = pixel_values
        return inputs


def register() -> None:
    AutoConfig.register("backward_pass_vlm", BackwardPassVLMConfig, exist_ok=True)
    AutoModelForImageTextToText.register(
        BackwardPassVLMConfig, BackwardPassVLMForConditionalGeneration, exist_ok=True
    )
