"""Section 19: the dataset format each TRL trainer expects, and a very short run of each.

The runs use a tiny random Llama with SmolLM2's tokenizer. They prove the data contract, not quality.
C6 trains real models with these losses.

Run: python c02-hf-stack/scripts/12_trl.py
"""

import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import AutoTokenizer, LlamaForCausalLM, PrinterCallback
from trl import DPOConfig, DPOTrainer, GRPOConfig, GRPOTrainer, SFTConfig, SFTTrainer

from hfstack.chat import SMOLLM2_TRAINING_TEMPLATE, assistant_labels_by_prefix
from hfstack.trl_runs import exact_answer_reward, preference_dataset, prompt_dataset, sft_dataset
from hfstack.training import clone_model
from hfstack.utils import CHAT_LM, be_polite, tiny_llama_config

be_polite()
torch.manual_seed(0)
tok = AutoTokenizer.from_pretrained(CHAT_LM)
tok.chat_template = SMOLLM2_TRAINING_TEMPLATE  # identical text, plus the {% generation %} markers
base = LlamaForCausalLM(
    tiny_llama_config(vocab_size=len(tok), bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
)
common = dict(
    max_steps=3, per_device_train_batch_size=4, logging_steps=1, report_to="none", save_strategy="no",
    use_cpu=True, bf16=False, gradient_checkpointing=False, disable_tqdm=True,
)


def losses(trainer) -> list[float]:
    return [round(e["loss"], 3) for e in trainer.state.log_history if "loss" in e and "train_loss" not in e]


print("--- formats ---")
print("SFT       ", json.dumps(sft_dataset(1)[0]))
print("preference", json.dumps(preference_dataset(1)[0]))
print("RL        ", json.dumps(prompt_dataset(1)[0]))

with tempfile.TemporaryDirectory() as tmp:
    print("\n--- SFTTrainer, assistant_only_loss=True ---")
    data = sft_dataset(32)
    start = time.perf_counter()
    args = SFTConfig(output_dir=tmp, assistant_only_loss=True, **common)
    trainer = SFTTrainer(model=clone_model(base), args=args, train_dataset=data, processing_class=tok)
    print("columns after preparation:", trainer.train_dataset.column_names)
    batch = trainer.data_collator([trainer.train_dataset[i] for i in range(2)])
    print("batch:", {k: tuple(v.shape) for k, v in batch.items()})
    ours = assistant_labels_by_prefix(tok, data[0]["messages"])
    print("TRL labels == our assistant_labels_by_prefix:", batch["labels"][0][: len(ours["labels"])].tolist() == ours["labels"])
    trainer.remove_callback(PrinterCallback)
    trainer.train()
    print(f"loss per step {losses(trainer)}, {time.perf_counter() - start:.1f} s")

    print("\n--- DPOTrainer ---")
    data = preference_dataset(32)
    start = time.perf_counter()
    args = DPOConfig(output_dir=tmp, **common)
    trainer = DPOTrainer(
        model=clone_model(base), ref_model=clone_model(base), args=args, train_dataset=data, processing_class=tok
    )
    print("columns after preparation:", trainer.train_dataset.column_names)
    trainer.remove_callback(PrinterCallback)
    trainer.train()
    print(f"loss per step {losses(trainer)}, {time.perf_counter() - start:.1f} s")
    print("ln 2 =", round(float(torch.log(torch.tensor(2.0))), 3), "(the DPO loss at step 0, when model == reference)")

    print("\n--- GRPOTrainer ---")
    data = prompt_dataset(32)
    start = time.perf_counter()
    args = GRPOConfig(output_dir=tmp, num_generations=4, max_completion_length=8, **common)
    trainer = GRPOTrainer(
        model=clone_model(base), reward_funcs=exact_answer_reward, args=args, train_dataset=data, processing_class=tok
    )
    print("columns:", trainer.train_dataset.column_names)
    trainer.remove_callback(PrinterCallback)
    trainer.train()
    print(f"loss per step {losses(trainer)}, {time.perf_counter() - start:.1f} s")
    first = trainer.state.log_history[0]
    print({k: first[k] for k in ("reward", "reward_std", "frac_reward_zero_std", "grad_norm")})
    print("reward test:", exact_answer_reward([[{"content": "42"}], [{"content": "It is 7"}]], ["42", "8"]))
