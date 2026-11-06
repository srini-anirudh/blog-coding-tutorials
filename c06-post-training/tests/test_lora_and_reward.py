"""LoRA against PEFT, merge, adapter size, and the pairwise reward loss."""

import torch
from peft import LoraConfig, get_peft_model
from torch import nn

from posttrain.lora import DEFAULT_TARGETS, LoRALinear, apply_lora, lora_state_dict, merge_lora
from posttrain.models import tiny_llama
from posttrain.reward_model import pairwise_loss
from posttrain.utils import count_parameters


def test_lora_starts_at_the_base_model():
    model = tiny_llama()
    ids = torch.randint(3, 100, (2, 12))
    with torch.no_grad():
        before = model(input_ids=ids).logits.clone()
        apply_lora(model, r=4)
        after = model(input_ids=ids).logits
    torch.testing.assert_close(before, after)


def test_only_adapter_weights_train():
    model = apply_lora(tiny_llama(), r=4)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable and all("lora_" in n for n in trainable)
    n_train, n_total = count_parameters(model)
    assert n_train == sum(v.numel() for v in lora_state_dict(model).values())
    assert n_train < n_total


def test_lora_linear_matches_peft_with_the_same_factors():
    """Copy PEFT's A and B into our layer: the whole model must then give the same logits."""
    ours = apply_lora(tiny_llama(seed=0), r=4, alpha=8)
    targets = [t for t in DEFAULT_TARGETS]
    theirs = get_peft_model(tiny_llama(seed=0), LoraConfig(r=4, lora_alpha=8, target_modules=targets, bias="none"))

    peft_layers = {n.replace("base_model.model.", ""): m for n, m in theirs.named_modules() if hasattr(m, "lora_A")}
    assert len(peft_layers) > 0
    torch.manual_seed(1)
    for name, module in ours.named_modules():
        if isinstance(module, LoRALinear):
            with torch.no_grad():
                module.lora_B.copy_(0.1 * torch.randn_like(module.lora_B))  # move B off zero
                peft_layers[name].lora_A["default"].weight.copy_(module.lora_A)
                peft_layers[name].lora_B["default"].weight.copy_(module.lora_B)
    ids = torch.randint(3, 100, (2, 12))
    ours.eval(), theirs.eval()
    with torch.no_grad():
        a, b = ours(input_ids=ids).logits, theirs(input_ids=ids).logits
    torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)
    # Same trainable count as PEFT, so the "trainable parameters" row of the comparison means the same thing.
    assert count_parameters(ours)[0] == theirs.get_nb_trainable_parameters()[0]


def test_merge_preserves_outputs_and_removes_the_wrappers():
    model = apply_lora(tiny_llama(), r=4)
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, LoRALinear):
                m.lora_B.add_(0.05)
    ids = torch.randint(3, 100, (2, 12))
    with torch.no_grad():
        before = model(input_ids=ids).logits.clone()
        merge_lora(model)
        after = model(input_ids=ids).logits
    torch.testing.assert_close(before, after, atol=1e-4, rtol=1e-4)
    assert not any(isinstance(m, LoRALinear) for m in model.modules())
    assert all(type(getattr(model.model.layers[0].self_attn, "q_proj")) is nn.Linear for _ in [0])


def test_pairwise_loss_by_hand():
    loss, acc = pairwise_loss(torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]))
    want = -(torch.log(torch.sigmoid(torch.tensor(1.0))) + torch.log(torch.sigmoid(torch.tensor(-1.0)))) / 2
    assert abs(loss.item() - want.item()) < 1e-6
    assert acc.item() == 0.5
