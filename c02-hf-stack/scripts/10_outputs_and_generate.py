"""Sections 16-17: what a forward pass returns, and generate() written out as a loop.

Run: python c02-hf-stack/scripts/10_outputs_and_generate.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from hfstack.generation import generate_with_cache, greedy_no_cache
from hfstack.inspection import describe_outputs
from hfstack.utils import BASE_LM, be_polite

be_polite()
tok = AutoTokenizer.from_pretrained(BASE_LM)
tok.pad_token = tok.eos_token
tok.padding_side = "left"  # generation appends on the right, so padding must stay on the left

# eager attention, because the SDPA kernel never materialises the attention weights.
model = AutoModelForCausalLM.from_pretrained(BASE_LM, dtype=torch.float32, attn_implementation="eager").eval()
config = model.config

enc = tok(["The capital of France is", "Gradient descent"], return_tensors="pt", padding=True)
print("input_ids", enc["input_ids"].tolist())
print("attention", enc["attention_mask"].tolist())

with torch.no_grad():
    out = model(**enc, labels=enc["input_ids"], output_hidden_states=True, output_attentions=True)
print(f"\n{type(out).__name__}")
for key, shape in describe_outputs(out).items():
    print(f"  {key:<16} {shape}")
print("\nhidden_states has num_hidden_layers + 1 entries:", len(out.hidden_states), "=", config.num_hidden_layers, "+ 1")
print("attentions row sums to 1:", out.attentions[0][1, 0, -1].sum().item())
print(
    "cached key shape: (B, num_key_value_heads, T, head_dim) =",
    (2, config.num_key_value_heads, enc["input_ids"].size(1), config.head_dim),
)

print("\n--- the KV cache grows by one position per step ---")
cache = DynamicCache(config=config)
ids, mask = enc["input_ids"], enc["attention_mask"]
step_input = ids
with torch.no_grad():
    for step in range(3):
        position_ids = ((mask.cumsum(-1) - 1).clamp(min=0))[:, -step_input.size(1) :]
        logits = model(
            input_ids=step_input, attention_mask=mask, position_ids=position_ids, past_key_values=cache, use_cache=True
        ).logits
        print(
            f"step {step}: input {tuple(step_input.shape)}, logits {tuple(logits.shape)}, "
            f"cache keys {tuple(cache.layers[0].keys.shape)}"
        )
        step_input = logits[:, -1].argmax(-1, keepdim=True)
        mask = torch.cat([mask, torch.ones_like(step_input)], dim=-1)

print("\n--- our loop against model.generate ---")
new = 12
reference = model.generate(**enc, max_new_tokens=new, do_sample=False, pad_token_id=tok.pad_token_id)
ours = generate_with_cache(
    model, enc["input_ids"], enc["attention_mask"], new, eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id
)
print("greedy, tokens identical:", torch.equal(reference, ours))
print(tok.batch_decode(ours[:, enc["input_ids"].size(1) :]))

knobs = dict(temperature=0.7, top_k=50, top_p=0.9)
torch.manual_seed(0)
reference = model.generate(**enc, max_new_tokens=new, do_sample=True, pad_token_id=tok.pad_token_id, **knobs)
torch.manual_seed(0)
ours = generate_with_cache(
    model,
    enc["input_ids"],
    enc["attention_mask"],
    new,
    do_sample=True,
    eos_token_id=tok.eos_token_id,
    pad_token_id=tok.pad_token_id,
    **knobs,
)
print("sampling with the same seed, tokens identical:", torch.equal(reference, ours))
print(tok.batch_decode(ours[:, enc["input_ids"].size(1) :]))

print("\n--- break it: right padding at generation time ---")
tok.padding_side = "right"
right = tok(["The capital of France is", "Gradient descent"], return_tensors="pt", padding=True)
tok.padding_side = "left"
wrong = model.generate(**right, max_new_tokens=6, do_sample=False, pad_token_id=tok.pad_token_id)
good = model.generate(**enc, max_new_tokens=6, do_sample=False, pad_token_id=tok.pad_token_id)
print("left padded  :", repr(tok.decode(good[1, enc["input_ids"].size(1) :])))
print("right padded :", repr(tok.decode(wrong[1, right["input_ids"].size(1) :])))
