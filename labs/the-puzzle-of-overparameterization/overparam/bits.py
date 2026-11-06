"""Counting bits instead of parameters: the VC bound, the capacity budget, and a compression bound.

Three pieces of arithmetic from the post, plus one small experiment:
  * the VC bound for an 11M-parameter, 20-layer network on 50,000 images (section 1),
  * the 3.6 bits-per-parameter capacity budget against a 10T-token corpus (section 11),
  * the Occam / compression bound R <= R_hat + sqrt((b ln 2 + ln(1/delta)) / (2n)) (section 12),
    computed for a real trained classifier as we compress it harder: raw float32, then quantized, then
    pruned, quantized and lzma-coded. The same ladder on random labels shows the bound telling the two apart.
"""

import lzma
import math

import numpy as np
import torch
from torch import nn


def vc_bound(W=11e6, L=20, n=50_000):
    """sqrt(VC / n) with VC = W L log2 W, the Theta(W L log W) bound for ReLU networks without constants."""
    vc = W * L * math.log2(W)
    return {"vc_dim": vc, "ratio": vc / n, "bound_term": math.sqrt(vc / n)}


def capacity_budget(params=7e9, tokens=10e12, bits_per_param=3.6, bits_per_token=2.0):
    """Information the weights can hold versus information in the corpus (Morris et al. 2025's 3.6 bits)."""
    cap = params * bits_per_param
    corpus = tokens * bits_per_token
    return {"capacity_bits": cap, "capacity_GB": cap / 8e9, "corpus_bits": corpus, "corpus_TB": corpus / 8e12,
            "corpus_over_capacity": corpus / cap,
            "tokens_where_they_match": cap / bits_per_token}


def occam_bound(train_err, bits, n, delta=0.05):
    """Test error bound for a classifier with a b-bit description, chosen after seeing n i.i.d. examples."""
    return train_err + math.sqrt((bits * math.log(2) + math.log(1 / delta)) / (2 * n))


# ---------------------------------------------------------------------------- the ladder experiment

def xor_task(n, d=10, seed=0):
    """Binary labels y = 1[x1 * x2 + 0.5 x3 > 0] on Gaussian inputs: nonlinear, but low-dimensional."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=g)
    return x, ((x[:, 0] * x[:, 1] + 0.5 * x[:, 2]) > 0).long()


def train_classifier(x, y, width=32, epochs=20, seed=0):
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(x.shape[1], width), nn.ReLU(), nn.Linear(width, width), nn.ReLU(),
                          nn.Linear(width, 2))
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        for idx in torch.randperm(len(x), generator=g).split(256):
            opt.zero_grad()
            nn.functional.cross_entropy(model(x[idx]), y[idx]).backward()
            opt.step()
    return model


@torch.no_grad()
def error(model, x, y):
    return float((model(x).argmax(1) != y).float().mean())


@torch.no_grad()
def compress(model, levels_bits, keep=1.0):
    """Prune all but the largest `keep` fraction of each tensor's entries, quantize the survivors uniformly to
    2^levels_bits levels per tensor, and lzma-code the integer codes. Returns the compressed model and its
    description length in bits (codes plus one float32 scale per tensor)."""
    new = type(model)(*[type(m)(m.in_features, m.out_features) if isinstance(m, nn.Linear) else m for m in model])
    new.load_state_dict(model.state_dict())
    payload, bits = b"", 0
    for p in new.parameters():
        flat = p.flatten()
        if keep < 1.0:
            k = max(1, int(round(keep * flat.numel())))
            cut = flat.abs().kthvalue(flat.numel() - k + 1).values
            flat = torch.where(flat.abs() >= cut, flat, torch.zeros_like(flat))
        half = 2 ** (levels_bits - 1) - 1
        scale = float(flat.abs().max()) / max(half, 1) or 1.0
        codes = torch.round(flat / scale).clamp(-half, half)
        p.copy_((codes * scale).view_as(p))
        payload += codes.to(torch.int8).numpy().tobytes()
        bits += 32
    bits += 8 * len(lzma.compress(payload, preset=9 | lzma.PRESET_EXTREME))
    return new, bits


def ladder(n=50_000, n_test=20_000, width=32, delta=0.05, random_labels=False, seed=0):
    """Bound rungs for one trained classifier. Every rung's bound charges ln(#rungs) extra for having
    picked the best rung after the fact (a union bound over the menu of compression settings)."""
    x, y = xor_task(n, seed=seed)
    xt, yt = xor_task(n_test, seed=seed + 1)
    if random_labels:
        g = torch.Generator().manual_seed(seed + 2)
        y, yt = torch.randint(0, 2, (n,), generator=g), torch.randint(0, 2, (n_test,), generator=g)
    model = train_classifier(x, y, width=width, seed=seed)
    W = sum(p.numel() for p in model.parameters())
    menu = [("float32, uncompressed", None, None), ("8-bit quantized + lzma", 8, 1.0),
            ("4-bit quantized + lzma", 4, 1.0), ("keep 30%, 4-bit + lzma", 4, 0.3),
            ("keep 20%, 3-bit + lzma", 3, 0.2)]
    delta_each = delta / len(menu)
    rows = [{"rung": "VC term sqrt(W L log2 W / n)", "bits": None, "train_err": error(model, x, y),
             "test_err": error(model, xt, yt), "bound": error(model, x, y) + vc_bound(W, 3, n)["bound_term"]}]
    for name, b, keep in menu:
        if b is None:
            m, bits = model, 32 * W
        else:
            m, bits = compress(model, b, keep)
        tr = error(m, x, y)
        rows.append({"rung": name, "bits": bits, "train_err": tr, "test_err": error(m, xt, yt),
                     "bound": occam_bound(tr, bits, n, delta_each)})
    return {"params": W, "n": n, "random_labels": random_labels, "rows": rows}
