"""Assemble a TinyVLM from two pretrained checkpoints that have never seen each other."""

from __future__ import annotations

import torch
import transformers
from transformers import AutoTokenizer, LlamaForCausalLM, SiglipVisionConfig, SiglipVisionModel

from .chat import IMAGE_TOKEN
from .configuration import VLMConfig
from .image_processing import TinyVLMImageProcessor
from .modeling import TinyVLM
from .processing import TinyVLMProcessor
from .projector import build_projector

VISION_ID = "google/siglip-base-patch16-224"
LM_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"


def build_tinyvlm(
    vision_id: str = VISION_ID,
    lm_id: str = LM_ID,
    projector_type: str = "mlp",
    pixel_shuffle_factor: int = 2,
    vision_feature_layer: int = -1,
    strategy: str = "resize",
    max_tiles: int = 6,
    dtype: torch.dtype = torch.float32,
) -> tuple[TinyVLM, TinyVLMProcessor]:
    """Pretrained SigLIP + fresh projector + pretrained SmolLM2, plus a processor that matches them."""
    transformers.logging.set_verbosity_error()  # the encoder-only load reports every skipped text weight
    tokenizer = AutoTokenizer.from_pretrained(lm_id)
    tokenizer.add_tokens([IMAGE_TOKEN], special_tokens=True)  # one new id for the placeholder

    # SigLIP's pooling head turns 196 patch tokens into one vector; a VLM wants all 196, so drop it.
    vision_config = SiglipVisionConfig.from_pretrained(vision_id)
    vision_config.vision_use_head = False
    vision = SiglipVisionModel.from_pretrained(vision_id, config=vision_config, dtype=dtype)
    lm = LlamaForCausalLM.from_pretrained(lm_id, dtype=dtype)
    # A multiple of 64 keeps the embedding matmul shapes friendly; the extra rows are never used.
    lm.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)

    config = VLMConfig(
        vision_config=vision.config,
        text_config=lm.config,
        image_token_id=tokenizer.convert_tokens_to_ids(IMAGE_TOKEN),
        projector_type=projector_type,
        pixel_shuffle_factor=pixel_shuffle_factor,
        vision_feature_layer=vision_feature_layer,
    )
    with torch.device("meta"):  # a skeleton with no memory and no random-init cost ...
        model = TinyVLM(config)
    model.vision, model.lm = vision, lm  # ... filled with the two pretrained halves
    model.projector = build_projector(projector_type, config.projector_in_dim, config.text_config.hidden_size).to(dtype)

    image_processor = TinyVLMImageProcessor(size=vision.config.image_size, strategy=strategy, max_tiles=max_tiles)
    processor = TinyVLMProcessor(image_processor, tokenizer, image_seq_len=config.image_seq_len)
    return model.eval(), processor
