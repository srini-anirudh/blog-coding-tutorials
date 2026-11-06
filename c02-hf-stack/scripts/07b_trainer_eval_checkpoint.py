"""Sections 12-13: what the raw loop did not have. Evaluation, checkpoints, callbacks and resume.

Run: python c02-hf-stack/scripts/07b_trainer_eval_checkpoint.py
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import LlamaForCausalLM, PrinterCallback, Trainer, TrainerCallback

from hfstack.collators import CausalLMCollator
from hfstack.data import random_token_dataset
from hfstack.training import LoopConfig, clone_model, training_arguments
from hfstack.utils import be_polite, tiny_llama_config

be_polite()
torch.manual_seed(0)
config = tiny_llama_config()
model = LlamaForCausalLM(config)
train = random_token_dataset(64, config.vocab_size)
held_out = random_token_dataset(16, config.vocab_size, seed=1)
collate = CausalLMCollator(pad_token_id=config.pad_token_id)
cfg = LoopConfig(max_steps=30, warmup_steps=3)


class EventLog(TrainerCallback):
    """Logging is a callback. So are early stopping and the W&B integration. They all see these events."""

    def __init__(self) -> None:
        self.events: list[str] = []

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        self.events.append(f"step {state.global_step:>2}: evaluate, eval_loss {metrics['eval_loss']:.4f}")

    def on_save(self, args, state, control, **kwargs):
        self.events.append(f"step {state.global_step:>2}: save")


def trainer_for(net, output_dir: str, **overrides) -> Trainer:
    trainer = Trainer(
        model=net,
        args=training_arguments(cfg, output_dir, **overrides),
        train_dataset=train,
        eval_dataset=held_out,
        data_collator=collate,
    )
    trainer.remove_callback(PrinterCallback)
    return trainer


with tempfile.TemporaryDirectory() as tmp:
    events = EventLog()
    uninterrupted = clone_model(model)
    trainer = trainer_for(
        uninterrupted, tmp, eval_strategy="steps", eval_steps=10, save_strategy="steps", save_steps=10, save_total_limit=2
    )
    trainer.add_callback(events)
    trainer.train()

    print("--- what the callback saw ---")
    print("\n".join(events.events))

    print("\n--- save_total_limit=2 keeps the newest two checkpoints ---")
    print(sorted(p.name for p in Path(tmp).iterdir() if p.is_dir()))
    print("inside checkpoint-20:", sorted(p.name for p in (Path(tmp) / "checkpoint-20").iterdir()))

    state = json.loads((Path(tmp) / "checkpoint-20" / "trainer_state.json").read_text())
    print("trainer_state.json: global_step", state["global_step"], "| log entries", len(state["log_history"]))

    print("\n--- resume: a new Trainer and a new model object, starting from checkpoint-20 ---")
    resumed = clone_model(model)
    trainer_for(resumed, tmp).train(resume_from_checkpoint=str(Path(tmp) / "checkpoint-20"))
    gap = max((a - b).abs().max().item() for a, b in zip(uninterrupted.parameters(), resumed.parameters()))
    print(f"max |weight difference| against the uninterrupted run: {gap:.2e}")
