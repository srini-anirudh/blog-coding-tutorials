"""Real weights: the decoder loads SmolLM2-135M and matches Hugging Face's logits. Downloads if not cached."""

import pytest
import torch

pytestmark = pytest.mark.network


def test_smollm2_matches_hf():
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM

    from inferlab.decoder import Decoder

    path = snapshot_download("HuggingFaceTB/SmolLM2-135M", allow_patterns=["*.json", "*.safetensors", "*.txt"])
    hf = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32).eval()
    m = Decoder.from_hf(path, max_len=64)
    m.allocate_cache(1)
    ids = torch.randint(0, 49_152, (1, 32), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        torch.testing.assert_close(m.forward(ids, 0, last_only=False), hf(ids).logits, rtol=1e-3, atol=1e-3)
