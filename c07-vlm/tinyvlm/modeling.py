"""TinyVLM: pretrained vision encoder + projector + pretrained language model, as one PreTrainedModel.

Data flow for a batch with n crops:
    pixel_values  n x 3 x 224 x 224
      -> vision          n x N_v x d_v          (SigLIP-base: 196 x 768)
      -> pixel shuffle   n x N_img x d_v*s^2    (s = 2: 49 x 3072)
      -> projector       n x N_img x d_lm       (576 for SmolLM2-135M)
    input_ids     B x N  -> token embeddings B x N x d_lm, placeholders overwritten by image features
      -> lm              logits B x N x V
"""

from __future__ import annotations

import torch
from transformers import LlamaForCausalLM, PreTrainedModel, SiglipVisionModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from .configuration import VLMConfig
from .merge import merge_masked_scatter
from .positions import position_ids_1d
from .projector import build_projector, pixel_shuffle


class TinyVLM(PreTrainedModel):
    config_class = VLMConfig
    config: VLMConfig
    base_model_prefix = "tinyvlm"
    main_input_name = "input_ids"
    input_modalities = ("image", "text")
    # SmolLM2 ties the output head to the input embeddings; tell save/load about it.
    _tied_weights_keys = {"lm.lm_head.weight": "lm.model.embed_tokens.weight"}
    _supports_sdpa = True

    def __init__(self, config: VLMConfig):
        super().__init__(config)
        self.vision = SiglipVisionModel(config.vision_config)
        self.projector = build_projector(config.projector_type, config.projector_in_dim, config.text_config.hidden_size)
        self.lm = LlamaForCausalLM(config.text_config)
        self.post_init()

    def get_input_embeddings(self):
        return self.lm.get_input_embeddings()

    def get_image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """n x 3 x H x W -> n x N_img x d_lm."""
        layer = self.config.vision_feature_layer
        out = self.vision(pixel_values=pixel_values.to(self.vision.dtype), output_hidden_states=layer != -1)
        features = out.last_hidden_state if layer == -1 else out.hidden_states[layer]  # n x N_v x d_v
        features = pixel_shuffle(features, self.config.pixel_shuffle_factor)
        return self.projector(features)

    def embed(self, input_ids: torch.Tensor, pixel_values: torch.Tensor | None = None) -> torch.Tensor:
        """B x N ids (+ crops) -> B x N x d_lm input embeddings with image features scattered in."""
        embeds = self.get_input_embeddings()(input_ids)
        if pixel_values is None or pixel_values.shape[0] == 0:
            # Otherwise the model would read one meaningless embedding N_img times and never complain.
            if (input_ids == self.config.image_token_id).any():
                raise ValueError("input_ids contain <image> placeholders, but no pixel_values were given")
            return embeds
        features = self.get_image_features(pixel_values)
        return merge_masked_scatter(embeds, input_ids, features, self.config.image_token_id)

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values=None,
        use_cache: bool | None = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """Returns .logits (B x N x V) and, if labels are given, .loss (mean over labels != -100)."""
        inputs_embeds = self.embed(input_ids, pixel_values)
        if position_ids is None and attention_mask is not None and attention_mask.dim() == 2 and past_key_values is None:
            position_ids = position_ids_1d(attention_mask)  # correct under left padding too
        return self.lm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=labels,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **generate_kwargs,
    ) -> torch.Tensor:
        """Build the multimodal embeddings once, then reuse the language model's own decoding loop.

        Returns only the new tokens, B x T_new (Hugging Face does this when given inputs_embeds).
        """
        inputs_embeds = self.embed(input_ids, pixel_values)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        return self.lm.generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask, **generate_kwargs)

    # ---- training helpers (used by C8) ----------------------------------------------------------
    def set_trainable(self, vision: bool = False, projector: bool = True, lm: bool = False) -> None:
        """Freeze or unfreeze the three parts. Stage-1 alignment trains the projector only."""
        for module, flag in ((self.vision, vision), (self.projector, projector), (self.lm, lm)):
            module.requires_grad_(flag)

    def parameter_counts(self) -> dict[str, int]:
        counts = {name: sum(p.numel() for p in m.parameters()) for name, m in self.named_children()}
        counts["trainable"] = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return counts
