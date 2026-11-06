"""Overhead of each abstraction: step time of the raw loop, Trainer and Accelerate on the same run.

Run: python c02-hf-stack/benchmarks/bench_loops.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import LlamaForCausalLM

from hfstack.accelerate_loop import train_accelerate
from hfstack.collators import CausalLMCollator
from hfstack.data import random_token_dataset
from hfstack.training import LoopConfig, clone_model, train_raw, train_with_trainer
from hfstack.utils import be_polite, tiny_llama_config

be_polite()
torch.manual_seed(0)
config = tiny_llama_config(hidden_size=128, intermediate_size=256, num_hidden_layers=4)
model = LlamaForCausalLM(config)
data = random_token_dataset(256, config.vocab_size)
collate = CausalLMCollator(pad_token_id=config.pad_token_id)
cfg = LoopConfig(max_steps=100, batch_size=8, grad_accum=1, warmup_steps=5)

print("4-layer Llama, hidden 128, batch 8, 100 steps, Apple M5 CPU, 2 threads")
for name, fn in {
    "raw loop": lambda: train_raw(clone_model(model), data, collate, cfg),
    "Trainer": lambda: train_with_trainer(clone_model(model), data, collate, cfg),
    "Accelerate": lambda: train_accelerate(clone_model(model), data, collate, cfg),
}.items():
    start = time.perf_counter()
    fn()
    seconds = time.perf_counter() - start
    print(f"  {name:<12} {seconds:>6.2f} s total, {1000 * seconds / cfg.max_steps:>6.1f} ms/step")
