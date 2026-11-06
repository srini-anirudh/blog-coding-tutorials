"""Section 5: what a tokenizer returns, and every special case that bites in training.

Run: python c02-hf-stack/scripts/02_tokenizer.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformers import AutoModelForCausalLM, AutoTokenizer

from hfstack.utils import BASE_LM, CHAT_LM, be_polite, tiny_llama_config

be_polite()
tok = AutoTokenizer.from_pretrained(BASE_LM)

# 1. Encode and decode. Tokens are not words: the leading space belongs to the token.
text = "The backward pass computes gradients."
ids = tok(text)["input_ids"]
print("input_ids :", ids)
print("tokens    :", tok.convert_ids_to_tokens(ids))
print("decoded   :", repr(tok.decode(ids)))

# 2. Special tokens. Base and Instruct checkpoints of the same model disagree.
print(f"\n{'':<10}{'bos':<16}{'eos':<16}{'pad':<16}{'pad side':<10}")
for name in (BASE_LM, CHAT_LM):
    t = AutoTokenizer.from_pretrained(name)
    print(
        f"{name.split('/')[1][-7:]:<10}{t.bos_token!s:<16}{t.eos_token!s:<16}{t.pad_token!s:<16}{t.padding_side:<10}"
    )
print("BOS added automatically:", ids[0] == tok.bos_token_id)

# 3. Padding and truncation on a batch.
tok.pad_token = tok.eos_token
batch = tok(["Hello world", "The backward pass computes gradients."], padding=True, return_tensors="pt")
print("\nright padding")
print(" input_ids     ", batch["input_ids"].tolist())
print(" attention_mask", batch["attention_mask"].tolist())
tok.padding_side = "left"
batch = tok(["Hello world", "The backward pass computes gradients."], padding=True, return_tensors="pt")
print("left padding")
print(" input_ids     ", batch["input_ids"].tolist())
print(" attention_mask", batch["attention_mask"].tolist())
tok.padding_side = "right"
print("truncation, max_length=3:", tok(text, truncation=True, max_length=3)["input_ids"])

# 4. Added tokens: the vocabulary grows, so the embedding matrix must grow with it.
print("\nlen(tokenizer) before:", len(tok))
added = tok.add_special_tokens({"additional_special_tokens": ["<tool_call>", "</tool_call>"]})
print("added:", added, "| len(tokenizer) after:", len(tok))
print("new token ids:", tok("<tool_call>x</tool_call>")["input_ids"])

model = AutoModelForCausalLM.from_config(tiny_llama_config(vocab_size=49152))
print("embedding before resize:", tuple(model.get_input_embeddings().weight.shape))
try:
    import torch

    model(input_ids=torch.tensor([[len(tok) - 1]]))
except IndexError as err:
    print("forward with a new token, before resize -> IndexError:", err)
model.resize_token_embeddings(len(tok))
print("embedding after resize :", tuple(model.get_input_embeddings().weight.shape))
print("lm_head after resize   :", tuple(model.get_output_embeddings().weight.shape))
print("config.vocab_size      :", model.config.vocab_size)
print("lm_head still tied     :", model.lm_head.weight is model.model.embed_tokens.weight)
