"""Every script runs end to end with --quick. These take a while, so they are marked slow."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

LAB = Path(__file__).resolve().parents[1]

OFFLINE = ["01_linear_collapse", "02_gradient_decay", "03_dying_relu", "04_smooth_gates", "06_sparsity"]


def _run(name, tmp_path, *extra):
    cmd = [sys.executable, str(LAB / "scripts" / f"{name}.py"), "--out", str(tmp_path), *extra]
    subprocess.run(cmd, check=True, capture_output=True, timeout=900)
    return json.loads((tmp_path / f"{name}.json").read_text())


@pytest.mark.slow
@pytest.mark.parametrize("name", OFFLINE)
def test_script_quick(name, tmp_path):
    extra = [] if name == "04_smooth_gates" else ["--quick"]
    out = _run(name, tmp_path, *extra)
    assert out["hardware"].startswith("hardware:")


@pytest.mark.slow
def test_ffn_script_quick_offline(tmp_path):
    out = _run("05_ffn_variants", tmp_path, "--quick", "--synthetic")
    assert set(out["lm"]["by_kind"]) == {"relu", "gelu", "relu2", "bilinear", "reglu", "geglu", "swiglu"}


@pytest.mark.network
def test_gpt2_tanh_gelu_reproduces_checkpoint():
    """Our gelu_tanh is exactly the 'gelu_new' GPT-2 was trained with: swapping it in changes no logit."""
    import torch
    from transformers import GPT2LMHeadModel

    sys.path.insert(0, str(LAB))
    from nonlin.activations import Activation

    model = GPT2LMHeadModel.from_pretrained("gpt2").eval()
    ids = torch.arange(100, 164).unsqueeze(0)
    with torch.no_grad():
        ref = model(ids).logits
        for blk in model.transformer.h:
            blk.mlp.act = Activation("gelu_tanh")
        new = model(ids).logits
    assert (new - ref).abs().max().item() < 1e-3


@pytest.mark.network
@pytest.mark.slow
def test_pretrained_swaps_quick(tmp_path):
    out = _run("07_pretrained_swaps", tmp_path, "--quick")
    assert out["gpt2"]["gelu_tanh"]["max_logit_change"] < 1e-3
