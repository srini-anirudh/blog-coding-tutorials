"""Native save/load must reproduce the model exactly, including the weight tie."""

import torch

from c04_helpers import random_ids, small_model
from tinyllm import load_checkpoint, save_checkpoint


def test_round_trip_reproduces_logits(tmp_path):
    model = small_model()
    save_checkpoint(model, tmp_path / "ckpt")
    loaded = load_checkpoint(tmp_path / "ckpt").eval()
    ids = random_ids(2, 12)
    assert torch.equal(model(ids).logits, loaded(ids).logits)
    assert loaded.config == model.config


def test_tied_weights_are_stored_once_and_retied(tmp_path):
    from safetensors.torch import load_file

    model = small_model()
    save_checkpoint(model, tmp_path / "ckpt")
    assert "lm_head.weight" not in load_file(tmp_path / "ckpt" / "model.safetensors")
    loaded = load_checkpoint(tmp_path / "ckpt")
    assert loaded.lm_head.weight is loaded.embed_tokens.weight


def test_untied_round_trip(tmp_path):
    model = small_model(tie_word_embeddings=False)
    save_checkpoint(model, tmp_path / "ckpt")
    loaded = load_checkpoint(tmp_path / "ckpt")
    assert torch.equal(loaded.lm_head.weight, model.lm_head.weight)
    assert loaded.lm_head.weight is not loaded.embed_tokens.weight


def test_training_resumes_identically(tmp_path):
    """Save mid-training, reload, take one more step: same weights as never stopping."""
    torch.manual_seed(0)
    ids = random_ids(4, 16)

    def step(model, opt):
        opt.zero_grad()
        model(ids, labels=ids).loss.backward()
        opt.step()

    model = small_model().train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    step(model, opt)
    save_checkpoint(model, tmp_path / "ckpt")
    torch.save(opt.state_dict(), tmp_path / "ckpt" / "optim.pt")
    step(model, opt)

    resumed = load_checkpoint(tmp_path / "ckpt").train()
    opt2 = torch.optim.AdamW(resumed.parameters(), lr=1e-3)
    opt2.load_state_dict(torch.load(tmp_path / "ckpt" / "optim.pt"))
    step(resumed, opt2)
    for p1, p2 in zip(model.parameters(), resumed.parameters()):
        torch.testing.assert_close(p1, p2)
