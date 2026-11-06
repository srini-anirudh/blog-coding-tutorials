import pytest
import torch

from toollab.toolformer import corpus, filter_call


@pytest.mark.network
def test_gpt2_keeps_the_useful_calculator_call():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
    model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2", dtype=torch.float32).eval()
    site = next(s for s in corpus() if s["tool"] == "Calculator")
    useful = filter_call(model, tok, site["before"], site["after"], *site["calls"]["useful"])
    irrelevant = filter_call(model, tok, site["before"], site["after"], *site["calls"]["irrelevant"])
    assert useful["gain"] > 0.5 > irrelevant["gain"]
