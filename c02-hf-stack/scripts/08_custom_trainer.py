"""Section 14: subclass Trainer. With z_loss_weight=0 the subclass must equal the default.

Run: python c02-hf-stack/scripts/08_custom_trainer.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import LlamaForCausalLM

from hfstack.collators import CausalLMCollator
from hfstack.custom_trainer import ZLossTrainer
from hfstack.data import random_token_dataset
from hfstack.training import LoopConfig, clone_model, train_with_trainer
from hfstack.utils import be_polite, tiny_llama_config

be_polite()
torch.manual_seed(0)
config = tiny_llama_config()
model = LlamaForCausalLM(config)
data = random_token_dataset(64, config.vocab_size)
collate = CausalLMCollator(pad_token_id=config.pad_token_id)
cfg = LoopConfig(max_steps=40)

default = train_with_trainer(clone_model(model), data, collate, cfg)
same = train_with_trainer(clone_model(model), data, collate, cfg, trainer_cls=ZLossTrainer)
with_z = train_with_trainer(clone_model(model), data, collate, cfg, trainer_cls=ZLossTrainer, z_loss_weight=1e-3)

print("logged keys:", sorted(same))
print("max |loss diff|, default vs subclass with z=0:", f"{max(abs(a - b) for a, b in zip(default['loss'], same['loss'])):.2e}")
print(f"\n{'step':>4} {'z=0 loss':>10} {'accuracy':>9} {'mean logZ':>10} | {'z=1e-3 loss':>12} {'mean logZ':>10}")
for step in (0, 9, 19, 29, 39):
    print(
        f"{step:>4} {same['loss'][step]:>10.4f} {same['token_accuracy'][step]:>9.4f} {same['mean_log_z'][step]:>10.3f} | "
        f"{with_z['loss'][step]:>12.4f} {with_z['mean_log_z'][step]:>10.3f}"
    )
