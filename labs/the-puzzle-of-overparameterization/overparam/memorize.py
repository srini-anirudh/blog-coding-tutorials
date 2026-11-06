"""Networks memorize anything: the Zhang et al. (2017) label-corruption dial, at small scale.

Train the same MLP, with the same optimizer and no explicit regularizer, on a CIFAR-10 subset whose labels
have been partly or fully replaced by uniform random ones (and once on pure Gaussian noise images). Record
when training accuracy first reaches 100% and what the test accuracy is. Only the labels change between runs.
"""

import time

import numpy as np
import torch
from torch import nn


def load_cifar(n_train, n_test):
    """The first n_train / n_test CIFAR-10 images from the Hugging Face Hub, flattened to 3072 floats and
    standardized per channel with training-set statistics. Needs the dataset (downloads once)."""
    from datasets import load_dataset

    ds = load_dataset("uoft-cs/cifar10")

    def to_arrays(split, n):
        part = ds[split].select(range(n))
        x = np.stack([np.asarray(im, dtype=np.float32) / 255.0 for im in part["img"]])   # n x 32 x 32 x 3
        return x, np.asarray(part["label"])

    x, y = to_arrays("train", n_train)
    xt, yt = to_arrays("test", n_test)
    mean, std = x.mean((0, 1, 2)), x.std((0, 1, 2))
    prep = lambda a: torch.from_numpy(((a - mean) / std).reshape(len(a), -1).astype(np.float32))
    return prep(x), torch.from_numpy(y), prep(xt), torch.from_numpy(yt)


def synthetic_clusters(n, n_test, d=32, k=10, seed=0):
    """Offline stand-in for tests: k Gaussian clusters in d dimensions."""
    g = torch.Generator().manual_seed(seed)
    centers = torch.randn(k, d, generator=g) * 1.5
    def draw(m):
        y = torch.randint(0, k, (m,), generator=g)
        return centers[y] + torch.randn(m, d, generator=g), y
    x, y = draw(n)
    xt, yt = draw(n_test)
    return x, y, xt, yt


def corrupt_labels(y, frac, k=10, seed=0):
    """Replace a `frac` fraction of the labels (chosen at random) with uniform random classes."""
    g = torch.Generator().manual_seed(seed)
    y = y.clone()
    idx = torch.randperm(len(y), generator=g)[: int(round(frac * len(y)))]
    y[idx] = torch.randint(0, k, (len(idx),), generator=g)
    return y


def gaussian_pixels(x, seed=0):
    """Images replaced by Gaussian noise with the same per-feature mean and std."""
    g = torch.Generator().manual_seed(seed)
    return x.mean(0) + x.std(0) * torch.randn(x.shape, generator=g)


def mlp(d_in, width, k=10, seed=0):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(d_in, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU(),
                         nn.Linear(width, k))


@torch.no_grad()
def accuracy(model, x, y, batch=2048):
    hits = sum((model(x[i:i + batch]).argmax(1) == y[i:i + batch]).sum().item() for i in range(0, len(x), batch))
    return hits / len(x)


def fit(x, y, xt, yt, width=512, lr=1e-3, batch=128, max_epochs=300, seed=0):
    """Adam, no weight decay, no dropout, no augmentation. Stops one epoch after training accuracy hits 100%.
    Returns the epoch and step at which it did (None if it never did), and the final accuracies."""
    model = mlp(x.shape[1], width, seed=seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    steps, hit, t0 = 0, None, time.time()
    for epoch in range(1, max_epochs + 1):
        model.train()
        for idx in torch.randperm(len(x), generator=g).split(batch):
            opt.zero_grad()
            nn.functional.cross_entropy(model(x[idx]), y[idx]).backward()
            opt.step()
            steps += 1
        if accuracy(model, x, y) == 1.0:
            hit = (epoch, steps)
            break
    return {"epochs_to_fit": hit[0] if hit else None, "steps_to_fit": hit[1] if hit else None,
            "train_acc": accuracy(model, x, y), "test_acc": accuracy(model, xt, yt),
            "params": sum(p.numel() for p in model.parameters()), "seconds": round(time.time() - t0, 1)}
