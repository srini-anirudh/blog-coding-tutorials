import torch

from mixlab import grads as G
from mixlab.model import make_model
from mixlab.tasks import IGNORE, build_pool, decode, encode, prompt_and_answer
from mixlab.train import evaluate, make_pools, mixture_loss, sample_batch, train


def test_labels_are_the_answer_shifted_by_one():
    pool = build_pool("add", 20, seed=0)
    for k in range(20):
        p, a = prompt_and_answer(pool, k)
        text = decode(p)
        x, y = text[:-1].split("+")
        assert decode(a) == f"{int(x) + int(y):03d}\n"
        lab = pool.labels[k]
        assert int(lab[len(p) - 1]) == pool.input_ids[k, len(p)]


def test_style_adds_a_supervised_preamble():
    pool = build_pool("rev", 5, seed=0, style=">")
    p, a = prompt_and_answer(pool, 0)
    assert decode(a)[0] == ">" and decode(a)[1:5] == decode(p)[1:5][::-1]


def test_encode_decode_roundtrip():
    assert decode(encode("R1234=>4321\n")) == "R1234=>4321\n"


def test_sampler_follows_weights():
    tr, _ = make_pools(("add", "copy"), n_train=100, n_eval=1)
    gen = torch.Generator().manual_seed(0)
    who = []
    for _ in range(50):
        who += sample_batch(tr, {"add": 0.8, "copy": 0.2}, 64, gen)[2]
    assert abs(who.count("add") / len(who) - 0.8) < 0.03


def test_token_and_example_losses_agree_for_equal_lengths():
    tr, _ = make_pools(("rev", "copy"), n_train=32, n_eval=1)
    model = make_model(hidden=16, heads=2)
    ids = torch.cat([tr["rev"].input_ids, tr["copy"].input_ids])
    lab = torch.cat([tr["rev"].labels, tr["copy"].labels])
    assert torch.allclose(mixture_loss(model, ids, lab, "token"), mixture_loss(model, ids, lab, "example"))


def test_token_mean_gradient_decomposes_by_token_share():
    tr, _ = make_pools(("long", "copy"), n_train=16, n_eval=1)
    model = make_model(hidden=16, heads=2)
    ids = torch.cat([tr["long"].input_ids, tr["copy"].input_ids])
    lab = torch.cat([tr["long"].labels, tr["copy"].labels])
    full = G.flat_grad(model, mixture_loss(model, ids, lab))
    n_long = (tr["long"].labels != IGNORE).sum()
    share = float(n_long / (lab != IGNORE).sum())
    parts = share * G.flat_grad(model, mixture_loss(model, ids[:16], lab[:16])) + (1 - share) * G.flat_grad(
        model, mixture_loss(model, ids[16:], lab[16:])
    )
    assert torch.allclose(full, parts, atol=1e-6)


def test_gram_is_psd_and_pcgrad_removes_conflict():
    g = [torch.tensor([1.0, 0.0]), torch.tensor([-1.0, 1.0])]
    K = G.gram(g)
    assert torch.allclose(K, K.T) and float(torch.linalg.eigvalsh(K).min()) >= -1e-6
    proj = G.pcgrad_pair(g[0], g[1])
    assert abs(float(proj.dot(g[1]))) < 1e-6
    assert torch.equal(G.pcgrad_pair(g[0], torch.tensor([1.0, 1.0])), g[0])


def test_first_order_prediction_holds_for_small_steps():
    tr, ev = make_pools(("add", "rev"), n_train=64, n_eval=64)
    model = make_model(hidden=16, heads=2)
    g = G.task_gradients(model, ev, n=64)
    res = G.one_step_check(model, ev, g["add"], eta=1e-3, n=64)
    for actual, pred in res.values():
        assert abs(actual - pred) <= 0.1 * abs(pred) + 1e-6


def test_training_reduces_loss():
    tr, ev = make_pools(("copy",), n_train=512, n_eval=128)
    model = make_model(hidden=16, heads=2)
    before = evaluate(model, ev)["copy"]["loss"]
    train(model, tr, {"copy": 1}, steps=60, batch=32)
    assert evaluate(model, ev)["copy"]["loss"] < before - 0.5
