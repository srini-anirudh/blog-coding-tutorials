"""Sections 12-13: the same run as a raw loop and through Trainer. The curves must coincide.

Also shows the classic gradient-accumulation bug, and saves the curves for the post's figure.

Run: python c02-hf-stack/scripts/07_raw_vs_trainer.py
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import AutoTokenizer, LlamaForCausalLM

from hfstack.chat import assistant_labels_by_prefix
from hfstack.collators import CausalLMCollator
from hfstack.data import make_arithmetic_rows, to_messages
from hfstack.training import LoopConfig, clone_model, train_raw, train_with_trainer
from hfstack.utils import CHAT_LM, be_polite, hardware_label, tiny_llama_config

be_polite()
torch.manual_seed(0)

tok = AutoTokenizer.from_pretrained(CHAT_LM)
features = [
    assistant_labels_by_prefix(tok, to_messages(row)["messages"]) for row in make_arithmetic_rows(256, seed=0)
]
collate = CausalLMCollator(pad_token_id=tok.pad_token_id)

# A from-scratch Llama with SmolLM2's vocabulary: random weights, so the curve has somewhere to go.
model = LlamaForCausalLM(tiny_llama_config(vocab_size=len(tok), hidden_size=128, intermediate_size=256))
cfg = LoopConfig(learning_rate=3e-3, batch_size=8, grad_accum=2, max_steps=60, warmup_steps=6)

start = time.perf_counter()
raw = train_raw(clone_model(model), features, collate, cfg)
raw_seconds = time.perf_counter() - start
start = time.perf_counter()
hf = train_with_trainer(clone_model(model), features, collate, cfg)
hf_seconds = time.perf_counter() - start
naive = train_raw(clone_model(model), features, collate, cfg, normalize="micro_mean")

print(f"{'step':>4} {'raw loss':>10} {'Trainer':>10} {'|diff|':>9} {'raw gnorm':>10} {'Trainer':>10} {'lr':>9}")
for step in (0, 1, 2, 5, 10, 20, 30, 40, 50, 59):
    print(
        f"{step:>4} {raw['loss'][step]:>10.6f} {hf['loss'][step]:>10.6f} "
        f"{abs(raw['loss'][step] - hf['loss'][step]):>9.1e} "
        f"{raw['grad_norm'][step]:>10.6f} {hf['grad_norm'][step]:>10.6f} {hf['lr'][step]:>9.6f}"
    )
diff = max(abs(a - b) for a, b in zip(raw["loss"], hf["loss"]))
gdiff = max(abs(a - b) for a, b in zip(raw["grad_norm"], hf["grad_norm"]))
print(f"\nmax |loss diff| over {cfg.max_steps} steps : {diff:.2e}")
print(f"max |grad norm diff|        : {gdiff:.2e}")
print(f"wall clock ({hardware_label()}): raw {raw_seconds:.1f} s, Trainer {hf_seconds:.1f} s")

ndiff = max(abs(a - b) for a, b in zip(naive["loss"], hf["loss"]))
ngdiff = max(abs(a - b) for a, b in zip(naive["grad_norm"], hf["grad_norm"]))
print("\nbreak it: average each micro-batch, then divide by grad_accum")
print(f"max |loss diff| vs Trainer  : {ndiff:.2e}")
print(f"max |grad norm diff|        : {ngdiff:.2e}")
print(f"final loss: naive {naive['loss'][-1]:.4f}, Trainer {hf['loss'][-1]:.4f}")

out = Path(__file__).resolve().parents[1] / "results" / "raw_vs_trainer.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps({"raw": raw, "trainer": hf, "naive": naive}, indent=None))
print(f"saved {out.name}")
