"""Section 2-4: the config, the model class, and how the Auto classes pick one.

Run: python c02-hf-stack/scripts/01_config_and_auto.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
)
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

from hfstack.inspection import config_fields
from hfstack.utils import BASE_LM, be_polite, count_parameters, tiny_llama_config

be_polite()

# 1. The config: a small JSON file that fully describes the architecture. No weights are downloaded.
config = AutoConfig.from_pretrained(BASE_LM)
print(f"type(config) = {type(config).__name__}")
for name, value in config_fields(config).items():
    print(f"  {name:<24} {value}")

# 2. Dispatch: config.model_type -> class name -> class.
print(f"\nmodel_type {config.model_type!r} -> {MODEL_FOR_CAUSAL_LM_MAPPING_NAMES[config.model_type]}")

# 3. The same config, three heads. Built from the config only, so weights are random.
tiny = tiny_llama_config(num_labels=3)
for auto_cls in (AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification):
    model = auto_cls.from_config(tiny)
    _, total = count_parameters(model)
    head = [n for n, _ in model.named_children()][-1]
    print(f"{auto_cls.__name__:<36} -> {type(model).__name__:<32} last child: {head:<8} params: {total:,}")

# 4. The shapes each head produces for the same input.
ids = torch.randint(3, tiny.vocab_size, (2, 7))
with torch.no_grad():
    print("\nAutoModel                    last_hidden_state", tuple(AutoModel.from_config(tiny)(ids).last_hidden_state.shape))
    print("AutoModelForCausalLM         logits           ", tuple(AutoModelForCausalLM.from_config(tiny)(ids).logits.shape))
    print("AutoModelForSequenceClass.   logits           ", tuple(AutoModelForSequenceClassification.from_config(tiny)(ids).logits.shape))

# 5. Load the real weights. dtype=float32 because the checkpoint is stored in bfloat16.
model = AutoModelForCausalLM.from_pretrained(BASE_LM, dtype=torch.float32)
trainable, total = count_parameters(model)
print(f"\n{type(model).__name__}: {total:,} parameters ({total / 1e6:.1f} M), dtype {model.dtype}")
print("lm_head tied to embed_tokens:", model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr())
