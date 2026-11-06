import torch

from agentcraft.masking import VOCAB, EpisodeConfig, TinyLM, losses, make_batch, objective, train, world_grammar


def batch(n=4):
    cfg = EpisodeConfig()
    return make_batch(n, cfg, world_grammar(cfg), torch.Generator().manual_seed(0))


def test_layout_and_mask():
    tokens, mask, answer_pos = batch()
    for i in range(len(tokens)):
        scored = [VOCAB[t] for t, m in zip(tokens[i].tolist(), mask[i].tolist()) if m]
        assert scored[0] == "search" and scored[3] == "answer" and scored[-1] == "<eos>" and len(scored) == 7
        key = scored[1]
        assert scored[4] == key
        seq = [VOCAB[t] for t in tokens[i].tolist()]
        fact = seq.index(key, seq.index("<obs>"))           # the fact inside the observation
        assert seq[fact + 1] == VOCAB[tokens[i, answer_pos[i]]]
    share = mask.sum() / (tokens != 0).sum()
    assert 0.04 < share < 0.08


def test_masked_objective_ignores_world_targets():
    tokens, mask, _ = batch()
    model = TinyLM()
    l_act, l_world, n_act, n_world = losses(model, tokens, mask)
    assert torch.allclose(objective(model, tokens, mask, masked=True), l_act)
    unmasked = objective(model, tokens, mask, masked=False)
    assert torch.allclose(unmasked, (l_act * n_act + l_world * n_world) / (n_act + n_world))


def test_short_training_runs():
    out = train(True, steps=20, seed=0, eval_every=20, n_eval=32, log=lambda s: None)
    assert 0.0 <= out["final"]["answer_acc"] <= 1.0
    out = train(False, steps=20, seed=0, eval_every=20, n_eval=32, log=lambda s: None)
    assert "cos_unmasked_vs_policy" in out["final"]
