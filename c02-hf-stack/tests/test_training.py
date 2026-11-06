"""Numerical equivalence: raw loop == Trainer == subclass == Accelerate. Plus resume."""

import tempfile

import pytest
import torch
from transformers import LlamaForCausalLM, PrinterCallback, Trainer

from hfstack.accelerate_loop import train_accelerate
from hfstack.collators import CausalLMCollator
from hfstack.custom_trainer import ZLossTrainer
from hfstack.data import random_token_dataset
from hfstack.training import (
    LoopConfig,
    clone_model,
    count_label_tokens,
    summed_token_loss,
    train_raw,
    train_with_trainer,
    training_arguments,
)
from hfstack.utils import tiny_llama_config

CFG = LoopConfig(max_steps=12, warmup_steps=2)


@pytest.fixture(scope="module")
def setup():
    torch.manual_seed(0)
    config = tiny_llama_config()
    model = LlamaForCausalLM(config)
    data = random_token_dataset(64, config.vocab_size)
    return model, data, CausalLMCollator(pad_token_id=config.pad_token_id)


def close(a, b, tol=1e-5):
    assert len(a) == len(b)
    assert max(abs(x - y) for x, y in zip(a, b)) < tol


def test_loss_matches_library_loss(setup):
    model, data, collate = setup
    batch = collate([data[i] for i in range(4)])
    out = model(**batch)
    ours = summed_token_loss(out.logits, batch["labels"]) / count_label_tokens(batch["labels"])
    assert ours.item() == pytest.approx(out.loss.item(), abs=1e-6)


def test_raw_loop_equals_trainer(setup):
    model, data, collate = setup
    raw = train_raw(clone_model(model), data, collate, CFG)
    hf = train_with_trainer(clone_model(model), data, collate, CFG)
    close(raw["loss"], hf["loss"])
    close(raw["grad_norm"], hf["grad_norm"])
    close(raw["lr"], hf["lr"], tol=1e-10)


def test_custom_trainer_with_zero_weight_equals_default(setup):
    model, data, collate = setup
    default = train_with_trainer(clone_model(model), data, collate, CFG)
    custom = train_with_trainer(clone_model(model), data, collate, CFG, trainer_cls=ZLossTrainer)
    close(default["loss"], custom["loss"])
    assert "token_accuracy" in custom and "mean_log_z" in custom


def test_z_loss_changes_the_run(setup):
    model, data, collate = setup
    plain = train_with_trainer(clone_model(model), data, collate, CFG, trainer_cls=ZLossTrainer)
    z = train_with_trainer(clone_model(model), data, collate, CFG, trainer_cls=ZLossTrainer, z_loss_weight=1e-2)
    assert z["loss"][0] > plain["loss"][0]


def test_accelerate_equals_raw(setup):
    model, data, collate = setup
    raw = train_raw(clone_model(model), data, collate, CFG)
    acc = train_accelerate(clone_model(model), data, collate, CFG)
    close(raw["loss"], acc["loss"])
    close(raw["grad_norm"], acc["grad_norm"])


def test_naive_accumulation_differs(setup):
    # Micro-batches have different token counts, so averaging each one reweights tokens.
    model, data, collate = setup
    good = train_raw(clone_model(model), data, collate, CFG)
    bad = train_raw(clone_model(model), data, collate, CFG, normalize="micro_mean")
    assert max(abs(a - b) for a, b in zip(good["loss"], bad["loss"])) > 1e-3


def test_gradient_flows_to_every_parameter(setup):
    model, data, collate = setup
    fresh = clone_model(model)
    fresh(**collate([data[i] for i in range(4)])).loss.backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in fresh.parameters())


def test_resume_from_checkpoint_matches_uninterrupted(setup):
    model, data, collate = setup
    cfg = LoopConfig(max_steps=12, warmup_steps=2)
    with tempfile.TemporaryDirectory() as tmp:

        def trainer_for(m, **overrides):
            t = Trainer(model=m, args=training_arguments(cfg, tmp, **overrides), train_dataset=data, data_collator=collate)
            t.remove_callback(PrinterCallback)
            return t

        full = clone_model(model)
        trainer_for(full).train()

        # Save at step 6, then restart from that checkpoint in a new Trainer with a new model object.
        first = trainer_for(clone_model(model), save_strategy="steps", save_steps=6)
        first.train()
        resumed = clone_model(model)
        trainer_for(resumed).train(resume_from_checkpoint=f"{tmp}/checkpoint-6")

    for a, b in zip(full.parameters(), resumed.parameters()):
        assert torch.allclose(a, b, atol=1e-5)
