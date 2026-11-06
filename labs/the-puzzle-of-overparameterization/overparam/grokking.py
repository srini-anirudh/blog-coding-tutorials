"""Grokking on (a + b) mod p, with a small MLP on CPU.

Power et al. (2022) used a small transformer; Liu et al. (Omnigrok, 2023) and others show an MLP on learned
embeddings groks too, and it trains in about a minute on a CPU. The model embeds a and b with one shared
table, concatenates the two vectors, and maps them through one hidden ReLU layer to p logits. Training is
full-batch AdamW with strong weight decay, as in the grokking papers.

Besides accuracy we log two progress measures. The weight norm is the quantity Varma et al. (2023) say weight
decay trades against: memorizing needs norm that grows with the number of examples. The Fourier concentration
is the share of the embedding table's power (along the token axis) in its 8 strongest frequencies. Nanda et
al. (2023) found the generalizing circuit uses a few such frequencies, so this rises as the circuit forms.
"""

import torch
from torch import nn


def mod_add_split(p=97, train_frac=0.3, seed=0):
    a, b = torch.meshgrid(torch.arange(p), torch.arange(p), indexing="ij")
    pairs = torch.stack([a.flatten(), b.flatten()], 1)
    labels = (pairs[:, 0] + pairs[:, 1]) % p
    perm = torch.randperm(len(pairs), generator=torch.Generator().manual_seed(seed))
    cut = int(train_frac * len(pairs))
    tr, va = perm[:cut], perm[cut:]
    return pairs[tr], labels[tr], pairs[va], labels[va]


class ModMLP(nn.Module):
    def __init__(self, p=97, d=128, hidden=512, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.embed = nn.Embedding(p, d)
        self.mlp = nn.Sequential(nn.Linear(2 * d, hidden), nn.ReLU(), nn.Linear(hidden, p))

    def forward(self, pairs):
        return self.mlp(self.embed(pairs).flatten(1))


@torch.no_grad()
def fourier_concentration(embed_weight, top=8):
    """Share of the embedding's power, over frequencies 1..p//2 (k and p-k merged, DC dropped), held by the
    `top` strongest frequencies. A random table spreads power evenly: about top / (p // 2)."""
    p = embed_weight.shape[0]
    centred = embed_weight - embed_weight.mean(0)
    power = torch.fft.rfft(centred, dim=0).abs().pow(2).sum(1)[1: p // 2 + 1]
    return float(power.sort(descending=True).values[:top].sum() / power.sum())


@torch.no_grad()
def accuracy(model, x, y):
    return float((model(x).argmax(1) == y).float().mean())


def train(p=97, train_frac=0.3, weight_decay=1.0, lr=1e-3, steps=20000, log_every=500, seed=0):
    """Full-batch AdamW. Returns the log rows and the first steps at which train / val accuracy reach 99%."""
    xtr, ytr, xva, yva = mod_add_split(p, train_frac, seed)
    model = ModMLP(p, seed=seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.98))
    rows, first = [], {"train": None, "val": None}
    for step in range(1, steps + 1):
        opt.zero_grad()
        nn.functional.cross_entropy(model(xtr), ytr).backward()
        opt.step()
        if step % log_every == 0 or step == 1:
            tr, va = accuracy(model, xtr, ytr), accuracy(model, xva, yva)
            norm = float(torch.sqrt(sum((q.detach() ** 2).sum() for q in model.parameters())))
            rows.append({"step": step, "train_acc": tr, "val_acc": va, "weight_norm": norm,
                         "fourier_top8": fourier_concentration(model.embed.weight)})
            for key, acc in (("train", tr), ("val", va)):
                if first[key] is None and acc >= 0.99:
                    first[key] = step
    return rows, first
