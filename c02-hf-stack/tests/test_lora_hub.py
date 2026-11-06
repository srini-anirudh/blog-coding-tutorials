"""LoRA against the hand-written formula; safetensors parsed by hand; shards round trip."""

import pytest
import torch
from peft import PeftModel
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, LlamaForCausalLM

from hfstack.hub import read_index, read_safetensors_header, read_tensor_by_hand
from hfstack.lora import ATTENTION_PROJECTIONS, add_lora, adapter_size_bytes, lora_shapes, manual_lora_forward, merge_matches
from hfstack.training import clone_model
from hfstack.utils import count_parameters, tiny_llama_config


@pytest.fixture(scope="module")
def base():
    torch.manual_seed(0)
    return LlamaForCausalLM(tiny_llama_config()).eval()


def test_only_lora_tensors_are_trainable(base):
    model = add_lora(clone_model(base), r=4, alpha=8)
    shapes = lora_shapes(model)
    assert all("lora_" in name for name, _ in shapes)
    # 2 layers x 4 projections x (A + B).
    assert len(shapes) == 2 * len(ATTENTION_PROJECTIONS) * 2
    trainable, total = count_parameters(model)
    assert trainable == sum(torch.Size(s).numel() for _, s in shapes) and trainable < total


def test_fresh_adapter_changes_nothing(base):
    model = add_lora(clone_model(base)).eval()
    ids = torch.randint(3, 512, (2, 6))
    assert torch.allclose(model(input_ids=ids).logits, base(input_ids=ids).logits)


def test_manual_formula_equals_peft(base):
    model = add_lora(clone_model(base), r=4, alpha=8).eval()
    layer = model.base_model.model.model.layers[0].self_attn.q_proj
    torch.nn.init.normal_(layer.lora_B["default"].weight, std=0.1)
    x = torch.randn(2, 5, layer.in_features)
    assert torch.allclose(layer(x), manual_lora_forward(layer, x), atol=1e-6)


def test_merge_preserves_logits(base):
    model = add_lora(clone_model(base), r=4, alpha=8).eval()
    for name, p in model.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(p, std=0.01)
    merged, gap = merge_matches(model, torch.randint(3, 512, (2, 6)))
    assert gap < 1e-4
    assert isinstance(merged, LlamaForCausalLM)


def test_adapter_checkpoint_round_trip(base, tmp_path):
    model = add_lora(clone_model(base), r=4, alpha=8).eval()
    for name, p in model.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(p, std=0.01)
    assert adapter_size_bytes(model, tmp_path) > 0
    assert all("lora_" in key for key in load_file(tmp_path / "adapter_model.safetensors"))
    reloaded = PeftModel.from_pretrained(clone_model(base), tmp_path).eval()
    ids = torch.randint(3, 512, (2, 6))
    assert torch.allclose(reloaded(input_ids=ids).logits, model(input_ids=ids).logits, atol=1e-6)


def test_safetensors_by_hand_equals_library(base, tmp_path):
    base.save_pretrained(tmp_path)
    path = tmp_path / "model.safetensors"
    header = read_safetensors_header(path)
    library = load_file(path)
    assert set(header) - {"__metadata__"} == set(library)
    assert "lm_head.weight" not in library  # tied, saved once
    name = "model.layers.0.self_attn.q_proj.weight"
    assert torch.equal(read_tensor_by_hand(path, name), library[name])


def test_sharded_round_trip(base, tmp_path):
    base.save_pretrained(tmp_path, max_shard_size="100KB")
    index = read_index(tmp_path)
    assert index is not None and len(set(index["weight_map"].values())) > 1
    loaded = AutoModelForCausalLM.from_pretrained(tmp_path).eval()
    ids = torch.randint(3, 512, (2, 6))
    assert torch.equal(loaded(input_ids=ids).logits, base(input_ids=ids).logits)
