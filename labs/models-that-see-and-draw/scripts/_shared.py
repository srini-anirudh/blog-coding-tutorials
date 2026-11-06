"""Shared setup for the scripts: import path, data splits, and cached small models.

Trained models are cached in `checkpoints/` (ignored by git) so later scripts can reuse the tokenizer, the judge
and the token priors. Delete the folder to retrain everything from scratch.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

LAB = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB))

from seedraw import data  # noqa: E402
from seedraw.priors import TokenPrior, train_prior  # noqa: E402
from seedraw.readers import Judge, train_judge  # noqa: E402
from seedraw.tokenizers import FSQ, Tokenizer, VectorQuantizer, tokenize, train_tokenizer  # noqa: E402

CKPT = LAB / "checkpoints"
N_TRAIN, N_TEST = 10_000, 2_000
TOKENIZER_STEPS = 2_000
PRIOR_STEPS = 3_000

torch.set_num_threads(min(8, torch.get_num_threads()))


def splits() -> tuple[dict, dict]:
    return data.shapes(N_TRAIN, seed=0), data.shapes(N_TEST, seed=1)


def make_tokenizer(kind: str) -> Tokenizer:
    if kind == "vq":
        return Tokenizer(VectorQuantizer(1024, 64))
    if kind == "vq-lowdim":
        return Tokenizer(VectorQuantizer(1024, 64, code_dim=8, normalize=True))
    if kind == "fsq":
        return Tokenizer(FSQ([8, 5, 5, 5], 64))
    raise ValueError(kind)


def _cached(name: str, build, train):
    CKPT.mkdir(exist_ok=True)
    path = CKPT / f"{name}.pt"
    model = build()
    if path.exists():
        model.load_state_dict(torch.load(path))
    else:
        train(model)
        torch.save(model.state_dict(), path)
    model.eval()
    return model


def get_tokenizer(kind: str, train_set: dict) -> Tokenizer:
    return _cached(f"tokenizer-{kind}", lambda: make_tokenizer(kind),
                   lambda m: train_tokenizer(m, train_set["images"], TOKENIZER_STEPS))


def get_judge(train_set: dict) -> Judge:
    def fit(m):
        m.load_state_dict(train_judge(train_set).state_dict())
    return _cached("judge", Judge, fit)


def get_prior(order: str, tok: Tokenizer, train_set: dict) -> TokenPrior:
    vocab = tok.quantizer.codebook

    def fit(m):
        tokens = tokenize(tok, train_set["images"])
        train_prior(m, tokens, train_set["shape"], PRIOR_STEPS)
    return _cached(f"prior-{order}", lambda: TokenPrior(vocab, 16, 4, order), fit)
