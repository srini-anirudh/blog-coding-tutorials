"""Tokenizer setup: the LM's tokenizer plus the tokens a VLM needs.

New tokens:
    <image>            placeholder, one per visual token; its embedding is overwritten by the projector
    <loc_0>..<loc_31>  grounding coordinates for the `tokens` grounding format

`offline_tokenizer()` builds a byte-level tokenizer in memory with the same special tokens, so the
unit tests run without downloading anything.
"""

from __future__ import annotations

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from .grounding import LOC_BINS, loc_tokens

IMAGE_TOKEN = "<image>"
IM_START, IM_END = "<|im_start|>", "<|im_end|>"


def add_vlm_tokens(tok, bins: int = LOC_BINS):
    """Register <image> and the location tokens as special tokens (never split, never merged)."""
    tok.add_special_tokens({"additional_special_tokens": [IMAGE_TOKEN, *loc_tokens(bins)]},
                           replace_extra_special_tokens=False)
    tok.padding_side = "right"
    return tok


@torch.no_grad()
def spread_new_rows(model, new_ids: list[int], seed: int = 0) -> None:
    """Give new tokens distinct embedding rows with the statistics of the old vocabulary.

    resize_token_embeddings(mean_resizing=True) puts every new row within about 1e-3 of the mean row,
    so the 32 <loc_k> tokens start out indistinguishable, in the input and (tied) in the output head.
    Here each row becomes the old mean plus noise with the old per-dimension std: 32 ordinary, different words.
    """
    emb = model.lm.get_input_embeddings().weight
    old = emb[: min(new_ids)].float()
    noise = torch.randn(len(new_ids), emb.shape[1], generator=torch.Generator().manual_seed(seed))
    emb[new_ids] = (old.mean(0) + noise.to(emb.device) * old.std(0)).to(emb.dtype)


def build_tokenizer(name: str = "HuggingFaceTB/SmolLM2-135M-Instruct", bins: int = LOC_BINS):
    if name == "offline":
        return offline_tokenizer(bins)
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = IM_END
    return add_vlm_tokens(tok, bins)


def offline_tokenizer(bins: int = LOC_BINS) -> PreTrainedTokenizerFast:
    """256 byte tokens + chat and VLM special tokens. Slow to train on, perfect for tests."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers

    byte_tokens = list(pre_tokenizers.ByteLevel.alphabet())
    vocab = {"<|endoftext|>": 0, IM_START: 1, IM_END: 2}
    for t in sorted(byte_tokens):
        vocab[t] = len(vocab)
    core = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token="<|endoftext|>"))
    core.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    core.decoder = decoders.ByteLevel()
    tok = PreTrainedTokenizerFast(tokenizer_object=core, bos_token=IM_START, eos_token=IM_END,
                                  unk_token="<|endoftext|>", pad_token=IM_END)
    tok.add_special_tokens({"additional_special_tokens": [IM_START, IM_END]})
    return add_vlm_tokens(tok, bins)
