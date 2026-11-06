"""The blind-preference check: can the preference be predicted without the image?

Two blind judges:

1. A text-only probe. A logistic regression over bag-of-words features of the
   prompt and the two answers. It never sees pixels. If it beats 50% on held-out
   pairs, the dataset leaks the label through text, and DPO can lower its loss
   by learning the leak instead of looking.

2. The VLM with the image replaced by a grey square. If its preference accuracy
   on grey images rises during training as fast as on real images, the model
   learned a language prior, not a visual fact.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

import torch

from .preference import PreferencePair


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class Vocab:
    def __init__(self, texts: list[str]):
        words = sorted({w for t in texts for w in _words(t)})
        self.index = {w: i for i, w in enumerate(words)}

    def __len__(self) -> int:
        return len(self.index)

    def bow(self, text: str) -> torch.Tensor:
        v = torch.zeros(len(self.index))
        for w in _words(text):
            if w in self.index:
                v[self.index[w]] += 1
        return v


def pair_features(vocab: Vocab, prompt: str, a: str, b: str) -> torch.Tensor:
    """phi(a) - phi(b), plus prompt-word x answer-word crosses so 'how many' can interact with '3'.

    Antisymmetric by construction: swapping a and b negates the features, so the
    probe cannot prefer whichever answer happens to be listed first.
    """
    da = vocab.bow(a) - vocab.bow(b)                       # (W,)
    cross = torch.outer(vocab.bow(prompt).clamp(max=1), da)  # (W, W)
    return torch.cat([da, cross.flatten()])


@dataclass
class ProbeResult:
    train_acc: float
    test_acc: float
    per_kind: dict[str, float]


def fit_text_probe(train: list[PreferencePair], test: list[PreferencePair], steps: int = 300,
                   l2: float = 1e-3, seed: int = 0) -> ProbeResult:
    """Train a logistic regression to say which of (a, b) is the chosen answer."""
    rng = random.Random(seed)
    vocab = Vocab([t for p in train for t in (p.prompt, p.chosen, p.rejected)])

    def featurize(pairs):
        xs, ys = [], []
        for p in pairs:
            flip = rng.random() < 0.5  # random order, so the label is 50/50 by construction
            a, b = (p.rejected, p.chosen) if flip else (p.chosen, p.rejected)
            xs.append(pair_features(vocab, p.prompt, a, b))
            ys.append(0.0 if flip else 1.0)  # 1 means "a is chosen"
        return torch.stack(xs), torch.tensor(ys)

    xtr, ytr = featurize(train)
    xte, yte = featurize(test)
    w = torch.zeros(xtr.shape[1], requires_grad=True)  # no bias: features are antisymmetric
    opt = torch.optim.LBFGS([w], max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(xtr @ w, ytr) + l2 * w.pow(2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        acc = lambda x, y: float(((x @ w > 0).float() == y).float().mean())
        per_kind = {}
        for kind in sorted({p.kind for p in test}):
            idx = [i for i, p in enumerate(test) if p.kind == kind]
            per_kind[kind] = acc(xte[idx], yte[idx])
        return ProbeResult(acc(xtr, ytr), acc(xte, yte), per_kind)
