import torch

from optlab import Adam, Hybrid, Muon
from optlab.common import split_for_muon, tiny_model
from optlab.train import train


def test_muon_split_keeps_embeddings_on_adamw():
    model = tiny_model(hidden=32, layers=1, heads=2, kv_heads=1, ffn=64, seq=16)
    hidden, rest = split_for_muon(model)
    assert all(p.ndim == 2 for p in hidden)
    assert any(p is model.embed_tokens.weight for p in rest)
    assert len(hidden) == 7  # q, k, v, o, gate, up, down


def test_short_training_run_lowers_loss_with_hybrid():
    def make(m):
        h, r = split_for_muon(m)
        return Hybrid(Muon(h, lr=1e-2, ns_dtype=torch.float32), Adam(r, lr=1e-2))

    out = train(make, steps=12, batch=4, seq=32, warmup=2, log_every=1)
    assert not out["diverged"]
    assert out["curve"][-1][1] < out["curve"][0][1]
    assert 4.0 < out["state_bytes_per_param"] < 8.0
