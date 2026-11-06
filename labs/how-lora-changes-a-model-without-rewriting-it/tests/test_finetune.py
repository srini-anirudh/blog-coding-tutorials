"""The fine-tuning loop of scripts/07_finetune.py on a tiny random Llama (offline), and on SmolLM2-135M (network)."""

import importlib.util
from pathlib import Path

import pytest
import torch

from loralab.data import reverse_words

LAB = Path(__file__).resolve().parent.parent


def load_script():
    spec = importlib.util.spec_from_file_location("finetune07", LAB / "scripts" / "07_finetune.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def tiny_llama():
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    cfg = LlamaConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, vocab_size=97, max_position_embeddings=64)
    return LlamaForCausalLM(cfg)


def test_reverse_words():
    assert reverse_words("The cat, 2 sat.") == "ehT tac, 2 tas."


@pytest.mark.parametrize("method,rank", [("full", None), ("lora", 4), ("dora", 4), ("qlora", 4)])
def test_run_trains_each_method(method, rank):
    ft = load_script()
    g = torch.Generator().manual_seed(0)
    rows = torch.randint(0, 97, (32, 16), generator=g)
    evals = {"target": rows[:4], "source": rows[4:8]}
    lr = 1e-3 if method == "full" else 1e-2
    out = ft.run(tiny_llama(), method, rank, lr, rows, evals, steps=6, batch=4, device="cpu", ckpt_every=3)
    assert out["after"]["target"] < out["before"]["target"]
    if method == "full":
        assert out["trainable"] == sum(p.numel() for p in tiny_llama().parameters())
    else:
        assert out["trainable"] < 0.2 * sum(p.numel() for p in tiny_llama().parameters())
    assert len(out["dM_dD_trace"]) == 2 * 2 * 2  # 2 checkpoints x 2 layers x (q, v)
    if method in ("lora", "dora"):
        assert abs(out["after_merge_target"] - out["after"]["target"]) < 1e-4


@pytest.mark.network
def test_smollm2_nf4_is_worse_but_close():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from loralab.data import MODEL_ID, mean_loss
    from loralab.lora import ALL_LINEAR
    from loralab.quant import quantize_model_linears
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    ids = tok(["The history of the city begins in the twelfth century, when a small market grew up beside the "
               "river crossing and the bishop granted it a charter."], return_tensors="pt").input_ids
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
    fp = mean_loss(model, ids)
    quantize_model_linears(model, ALL_LINEAR)
    q = mean_loss(model, ids)
    assert fp < q < fp + 1.5
