"""bf16: the model must run in bf16 and stay close to float32."""

import torch

from c04_helpers import random_ids, small_model


def test_bf16_weights_run_and_stay_close():
    model = small_model()
    ids = random_ids(1, 16)
    fp32 = model(ids).logits
    bf16 = model.to(torch.bfloat16)(ids).logits
    assert bf16.dtype == torch.bfloat16
    assert (bf16.float() - fp32).abs().max() < 0.05


def test_bf16_cache_matches_full_forward():
    model = small_model().to(torch.bfloat16)
    ids = random_ids(1, 10)
    full = model(ids).logits
    out = model(ids[:, :6], use_cache=True)
    step = model(ids[:, 6:], past_key_values=out.past_key_values).logits
    assert (torch.cat((out.logits, step), 1).float() - full.float()).abs().max() < 0.05
    assert out.past_key_values.keys[0].dtype == torch.bfloat16
