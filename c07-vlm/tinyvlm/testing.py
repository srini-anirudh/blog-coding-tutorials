"""A tiny random TinyVLM and processor that need no downloads. Used by the tests (here and in C8).

Same architecture classes as the real build, only small: a 2-layer SigLIP on 32 x 32 crops with
8 x 8 patches (16 patch tokens), a 2-layer Llama with d = 48, and a character-level tokenizer.
"""

from __future__ import annotations

import string

import torch
from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers
from transformers import LlamaConfig, PreTrainedTokenizerFast, SiglipVisionConfig

from .chat import IM_END, IM_START, IMAGE_TOKEN
from .configuration import VLMConfig
from .image_processing import TinyVLMImageProcessor
from .modeling import TinyVLM
from .processing import TinyVLMProcessor

SPECIALS = ["<pad>", IM_START, IM_END, IMAGE_TOKEN]


def tiny_tokenizer() -> PreTrainedTokenizerFast:
    """One token per character plus the four special tokens."""
    vocab = {tok: i for i, tok in enumerate(SPECIALS + list(string.printable))}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="<pad>"))
    backend.pre_tokenizer = pre_tokenizers.Split(Regex(r"[\s\S]"), behavior="isolated")
    backend.decoder = decoders.Fuse()  # join characters back without spaces
    backend.add_special_tokens(SPECIALS)
    return PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="<pad>", eos_token=IM_END, unk_token="<pad>")


def tiny_vlm(
    projector_type: str = "mlp", pixel_shuffle_factor: int = 1, strategy: str = "resize", seed: int = 0
) -> tuple[TinyVLM, TinyVLMProcessor]:
    torch.manual_seed(seed)
    tokenizer = tiny_tokenizer()
    vision_config = SiglipVisionConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
        image_size=32, patch_size=8, vision_use_head=False,
    )
    text_config = LlamaConfig(
        vocab_size=128, hidden_size=48, intermediate_size=96, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, max_position_embeddings=512, tie_word_embeddings=True,
        pad_token_id=0, eos_token_id=tokenizer.convert_tokens_to_ids(IM_END), bos_token_id=None,
    )
    config = VLMConfig(
        vision_config, text_config, image_token_id=tokenizer.convert_tokens_to_ids(IMAGE_TOKEN),
        projector_type=projector_type, pixel_shuffle_factor=pixel_shuffle_factor,
    )
    model = TinyVLM(config).eval()
    image_processor = TinyVLMImageProcessor(size=32, strategy=strategy, max_tiles=4)
    return model, TinyVLMProcessor(image_processor, tokenizer, image_seq_len=config.image_seq_len)
