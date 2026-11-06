"""VLMConfig: one Hugging Face config that nests the vision config and the text config."""

from __future__ import annotations

from transformers import LlamaConfig, PretrainedConfig, SiglipVisionConfig


class VLMConfig(PretrainedConfig):
    """Everything needed to rebuild a TinyVLM with random weights; saved as config.json.

    Fields beyond the two sub-configs:
        image_token_id        id of the <image> placeholder in the tokenizer
        projector_type        "linear" or "mlp"
        pixel_shuffle_factor  1 keeps all N_v patch tokens; 2 folds 2 x 2 blocks, N_v / 4 tokens
        vision_feature_layer  which encoder layer to read: -1 is the last, -2 the one before it
    """

    model_type = "tinyvlm"
    sub_configs = {"vision_config": SiglipVisionConfig, "text_config": LlamaConfig}

    def __init__(
        self,
        vision_config: SiglipVisionConfig | dict | None = None,
        text_config: LlamaConfig | dict | None = None,
        image_token_id: int = 0,
        projector_type: str = "mlp",
        pixel_shuffle_factor: int = 1,
        vision_feature_layer: int = -1,
        **kwargs,
    ):
        # from_pretrained hands us plain dicts read from config.json; rebuild the typed configs.
        if isinstance(vision_config, dict):
            vision_config = SiglipVisionConfig(**vision_config)
        if isinstance(text_config, dict):
            text_config = LlamaConfig(**text_config)
        self.vision_config = vision_config if vision_config is not None else SiglipVisionConfig()
        self.text_config = text_config if text_config is not None else LlamaConfig()
        self.image_token_id = image_token_id
        self.projector_type = projector_type
        self.pixel_shuffle_factor = pixel_shuffle_factor
        self.vision_feature_layer = vision_feature_layer
        kwargs.setdefault("tie_word_embeddings", self.text_config.tie_word_embeddings)
        super().__init__(**kwargs)

    @property
    def patches_per_crop(self) -> int:
        """N_v: patch tokens the encoder emits for one crop, e.g. (224 / 16)^2 = 196."""
        return (self.vision_config.image_size // self.vision_config.patch_size) ** 2

    @property
    def image_seq_len(self) -> int:
        """N_img: image tokens the language model sees per crop, after pixel shuffle."""
        return self.patches_per_crop // (self.pixel_shuffle_factor**2)

    @property
    def projector_in_dim(self) -> int:
        return self.vision_config.hidden_size * self.pixel_shuffle_factor**2
