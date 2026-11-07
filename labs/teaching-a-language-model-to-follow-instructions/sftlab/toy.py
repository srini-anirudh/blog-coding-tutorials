"""A tiny offline tokenizer and model, so tests run with no download.

The tokenizer is byte-level (one token per byte, plus three special tokens) and carries a ChatML
template. The model is a 2-layer Llama with a few hundred thousand parameters, randomly initialised.
"""

from __future__ import annotations

import torch
from tokenizers import Tokenizer, models, pre_tokenizers, decoders
from tokenizers.pre_tokenizers import ByteLevel
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

CHATML = (
    "{% if messages[0]['role'] != 'system' %}<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n{% endif %}"
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


def toy_tokenizer() -> PreTrainedTokenizerFast:
    alphabet = ByteLevel.alphabet()
    vocab = {c: i for i, c in enumerate(sorted(alphabet))}
    tk = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    tk.add_special_tokens(["<|im_start|>", "<|im_end|>", "<|pad|>"])
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|im_end|>", pad_token="<|pad|>",
                                  bos_token="<|im_start|>")
    tok.chat_template = CHATML
    return tok


def toy_model(tok, seed: int = 0) -> LlamaForCausalLM:
    torch.manual_seed(seed)
    cfg = LlamaConfig(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=2048,
                      pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id, tie_word_embeddings=True)
    return LlamaForCausalLM(cfg).eval()
