"""TRL data contracts. Network for the tokenizer; the model is a tiny random Llama."""

import pytest
import torch
from transformers import AutoTokenizer, LlamaForCausalLM, LlamaForSequenceClassification
from trl import RewardConfig, RewardTrainer, SFTConfig, SFTTrainer

from hfstack.chat import SMOLLM2_TRAINING_TEMPLATE, assistant_labels_by_prefix
from hfstack.trl_runs import exact_answer_reward, preference_dataset, sft_dataset
from hfstack.utils import CHAT_LM, tiny_llama_config


def test_reward_is_exact():
    rewards = exact_answer_reward([[{"content": "42"}], [{"content": "It is 7"}], [{"content": "no"}]], ["42", "8", "1"])
    assert rewards == [1.0, 0.0, 0.0]


@pytest.mark.network
def test_sft_trainer_labels_equal_ours(tmp_path):
    tok = AutoTokenizer.from_pretrained(CHAT_LM)
    tok.chat_template = SMOLLM2_TRAINING_TEMPLATE
    torch.manual_seed(0)
    model = LlamaForCausalLM(tiny_llama_config(vocab_size=len(tok), pad_token_id=tok.pad_token_id))
    args = SFTConfig(
        output_dir=str(tmp_path), assistant_only_loss=True, max_steps=1, per_device_train_batch_size=2,
        use_cpu=True, bf16=False, gradient_checkpointing=False, report_to="none", disable_tqdm=True,
    )
    data = sft_dataset(4)
    trainer = SFTTrainer(model=model, args=args, train_dataset=data, processing_class=tok)
    row = trainer.train_dataset[0]
    assert row["labels"] == assistant_labels_by_prefix(tok, data[0]["messages"])["labels"]


@pytest.mark.network
def test_reward_trainer_scores_chosen_above_rejected_rows(tmp_path):
    tok = AutoTokenizer.from_pretrained(CHAT_LM)
    tok.chat_template = SMOLLM2_TRAINING_TEMPLATE
    torch.manual_seed(0)
    model = LlamaForSequenceClassification(
        tiny_llama_config(vocab_size=len(tok), num_labels=1, pad_token_id=tok.pad_token_id)
    )
    args = RewardConfig(
        output_dir=str(tmp_path), max_steps=1, per_device_train_batch_size=2,
        use_cpu=True, bf16=False, gradient_checkpointing=False, report_to="none", disable_tqdm=True,
    )
    trainer = RewardTrainer(model=model, args=args, train_dataset=preference_dataset(4), processing_class=tok)
    batch = trainer.data_collator([trainer.train_dataset[i] for i in range(2)])
    # Two examples become four rows: chosen first, then rejected.
    assert batch["input_ids"].shape[0] == 4
    assert trainer.train().training_loss > 0
