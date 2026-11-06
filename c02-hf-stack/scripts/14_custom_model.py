"""Section 22: a custom architecture, registered with the Auto classes, saved, loaded, trained, generating.

Run: python c02-hf-stack/scripts/14_custom_model.py
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from hfstack.collators import CausalLMCollator
from hfstack.custom_model import BackwardPassConfig, BackwardPassForCausalLM, register
from hfstack.data import random_token_dataset
from hfstack.generation import greedy_no_cache
from hfstack.hub import list_files, read_safetensors_header
from hfstack.inspection import describe_outputs
from hfstack.lora import add_lora
from hfstack.training import LoopConfig, clone_model, train_raw, train_with_trainer
from hfstack.utils import be_polite, count_parameters

be_polite()
torch.manual_seed(0)

config = BackwardPassConfig()
model = BackwardPassForCausalLM(config).eval()

print("--- what PreTrainedModel adds on top of nn.Module ---")
print(f"  {'method':<32} {'nn.Module':<10} BackwardPassForCausalLM")
for method in (
    "forward", "state_dict", "save_pretrained", "from_pretrained", "push_to_hub", "generate",
    "resize_token_embeddings", "gradient_checkpointing_enable", "tie_weights",
):
    print(f"  {method:<32} {hasattr(nn.Module, method)!s:<10} {hasattr(model, method)}")
print(f"  post_init drew embeddings with std {model.model.embed_tokens.weight.std().item():.4f}, "
      f"config.initializer_range = {config.initializer_range}")

print("\n--- before register(): the Auto classes do not know the model type ---")
with tempfile.TemporaryDirectory() as tmp:
    model.save_pretrained(tmp)
    try:
        AutoConfig.from_pretrained(tmp)
    except (ValueError, KeyError) as err:
        print(f"{type(err).__name__}: {str(err)[:110]}...")

    register()
    print("\n--- after register() ---")
    for name, size in list_files(tmp):
        print(f"  {name:<26} {size:>9,}")
    print("  architectures:", json.loads((Path(tmp) / "config.json").read_text())["architectures"])
    header = read_safetensors_header(Path(tmp) / "model.safetensors")
    print("  lm_head saved separately:", "lm_head.weight" in header)

    print("AutoConfig           ->", type(AutoConfig.from_pretrained(tmp)).__name__)
    print("AutoModel            ->", type(AutoModel.from_pretrained(tmp)).__name__)
    loaded = AutoModelForCausalLM.from_pretrained(tmp).eval()
    print("AutoModelForCausalLM ->", type(loaded).__name__)

    ids = torch.randint(3, config.vocab_size, (2, 7))
    with torch.no_grad():
        same = torch.equal(model(input_ids=ids).logits, loaded(input_ids=ids).logits)
    print("save/load round trip, logits identical:", same)
    print("tied after load:", loaded.lm_head.weight is loaded.model.embed_tokens.weight)

print("\n--- outputs ---")
with torch.no_grad():
    out = model(input_ids=ids, labels=ids, output_hidden_states=True)
for key, shape in describe_outputs(out).items():
    print(f"  {key:<16} {shape}")

print("\n--- generate(), from GenerationMixin ---")
tokens = model.generate(ids, max_new_tokens=8, do_sample=False)
print("generate equals a no-cache greedy loop:", torch.equal(tokens, greedy_no_cache(model, ids, 8)))

print("\n--- Trainer treats it like any other model ---")
data = random_token_dataset(64, config.vocab_size)
collate = CausalLMCollator(pad_token_id=config.pad_token_id)
cfg = LoopConfig(max_steps=20)
fresh = BackwardPassForCausalLM(config)
raw = train_raw(clone_model(fresh), data, collate, cfg)
hf = train_with_trainer(clone_model(fresh), data, collate, cfg)
print(f"raw vs Trainer, max |loss diff|: {max(abs(a - b) for a, b in zip(raw['loss'], hf['loss'])):.2e}")

print("\n--- PEFT needs only module names ---")
peft_model = add_lora(clone_model(fresh), r=4, alpha=8, targets=["qkv_proj", "o_proj"])
print("trainable, total:", count_parameters(peft_model))
