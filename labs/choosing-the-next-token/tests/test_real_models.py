"""Checks on the real SmolLM2 models. They download the weights if the Hub cache is empty, hence `network`."""

import pytest
import torch

from nexttoken import lm, speculative

pytestmark = pytest.mark.network


@pytest.fixture(scope="module")
def smol():
    return lm.load(lm.BASE)


def test_decode_matches_hf_generate_greedy(smol):
    tok, model = smol
    ids = tok("The history of the city begins", return_tensors="pt").input_ids
    ours = lm.decode(model, ids, lambda z, h: z.argmax(-1), 20)
    ref = model.generate(ids, max_new_tokens=20, do_sample=False, pad_token_id=tok.eos_token_id)[:, ids.shape[1]:]
    assert torch.equal(ours, ref)


def test_greedy_speculative_matches_target_on_real_models(smol):
    tok, draft = smol
    _, target = lm.load(lm.LARGE)
    ids = tok("In 1850, the railway reached the town and", return_tensors="pt").input_ids
    spec, stats = speculative.speculative_generate(draft, target, ids, 24, gamma=4, temperature=0)
    assert spec == lm.decode(target, ids, lambda z, h: z.argmax(-1), 24)[0].tolist()
    assert stats["target_calls"] < 24
