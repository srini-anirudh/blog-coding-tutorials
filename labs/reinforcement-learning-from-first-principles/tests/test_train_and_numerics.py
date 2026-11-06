"""A short training run, and the sampler-vs-trainer comparison on a tiny random model and on SmolLM2-135M."""

import pytest
import torch

from rlfp.numerics import decode_logprobs, forward_logprobs, mismatch_stats
from rlfp.train import TrainConfig, train


@pytest.mark.parametrize("method", ["rloo", "ppo", "critic_gae"])
def test_training_improves_exact_pass_rate(method):
    out = train(TrainConfig(method=method, steps=40, lr=1e-2, eval_every=40))
    hist = out["history"]
    assert hist[-1]["pass@1"] > hist[0]["pass@1"] + 0.1


def _tiny_llama():
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2)
    return LlamaForCausalLM(cfg).eval()


def test_decode_matches_forward_in_float32_and_bf16_differs():
    model = _tiny_llama()
    seqs = torch.randint(0, 128, (3, 20), generator=torch.Generator().manual_seed(0))
    a, b = decode_logprobs(model, seqs, 5), forward_logprobs(model, seqs, 5)
    assert a.shape == (3, 15) and torch.allclose(a, b, atol=1e-4)
    m16 = model.to(torch.bfloat16)
    st = mismatch_stats(forward_logprobs(m16, seqs, 5), b, lengths=(15,))
    assert st["token_sigma"] > 1e-5 and st["seq_std_T15"] > 0


@pytest.mark.network
def test_smollm2_sampler_and_trainer_agree_in_float32():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = "HuggingFaceTB/SmolLM2-135M"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32).eval()
    seqs = torch.tensor([tok("The capital of France is Paris, and the capital of Italy is Rome.").input_ids])
    a, b = decode_logprobs(model, seqs, 4), forward_logprobs(model, seqs, 4)
    assert torch.allclose(a, b, atol=1e-3)
