"""Section 15: the Accelerate loop on one process, and under DDP with two CPU processes.

Single process:  python c02-hf-stack/scripts/09_accelerate.py
Two processes:   torchrun --nproc_per_node 2 c02-hf-stack/scripts/09_accelerate.py --ddp
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from accelerate import Accelerator
from transformers import LlamaForCausalLM

from hfstack.accelerate_loop import train_accelerate
from hfstack.collators import CausalLMCollator
from hfstack.data import random_token_dataset
from hfstack.training import LoopConfig, clone_model, train_raw
from hfstack.utils import be_polite, tiny_llama_config

parser = argparse.ArgumentParser()
parser.add_argument("--ddp", action="store_true", help="run under torchrun; prints rank-0 history")
parser.add_argument("--save", type=str, default=None, help="write the loss history to this JSON file")
args = parser.parse_args()

be_polite(threads=1)
torch.manual_seed(0)
config = tiny_llama_config()
model = LlamaForCausalLM(config)
data = random_token_dataset(64, config.vocab_size)
collate = CausalLMCollator(pad_token_id=config.pad_token_id)

accelerator = Accelerator(cpu=True)
# One write per process, newline included, so the two ranks do not interleave within a line.
print(
    f"[rank {accelerator.process_index}] processes={accelerator.num_processes} "
    f"backend={accelerator.state.backend} device={accelerator.device}\n",
    end="",
    flush=True,
)

if args.ddp:
    # Two processes with batch 4 see the same global batch as one process with batch 8.
    cfg = LoopConfig(batch_size=4, grad_accum=2, max_steps=20)
    single = LoopConfig(batch_size=8, grad_accum=2, max_steps=20)
    history = train_accelerate(clone_model(model), data, collate, cfg, accelerator)
    if accelerator.is_main_process:
        reference = train_raw(clone_model(model), data, collate, single)
        diff = max(abs(a - b) for a, b in zip(history["loss"], reference["loss"]))
        gdiff = max(abs(a - b) for a, b in zip(history["grad_norm"], reference["grad_norm"]))
        print(f"2 processes x batch 4 vs 1 process x batch 8: max |loss diff| {diff:.2e}, max |grad norm diff| {gdiff:.2e}")
else:
    cfg = LoopConfig()
    history = train_accelerate(clone_model(model), data, collate, cfg, accelerator)
    reference = train_raw(clone_model(model), data, collate, cfg)
    diff = max(abs(a - b) for a, b in zip(history["loss"], reference["loss"]))
    print(f"Accelerate vs raw loop, 1 process: max |loss diff| {diff:.2e}")
    print("first losses:", [round(x, 5) for x in history["loss"][:4]])

if args.save and accelerator.is_main_process:
    Path(args.save).write_text(json.dumps(history))
