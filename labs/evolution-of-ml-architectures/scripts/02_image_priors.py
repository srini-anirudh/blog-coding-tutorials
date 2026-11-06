"""The same image through three architectures: what a convolution's prior buys, measured.

1. Parameter counts: a 224x224x3 dense layer, and the 16x16 lab's 32,768 dense weights against 9 shared ones.
2. Receptive field of L stacked 3x3 convolutions, measured by gradient: 1 + L(k-1).
3. Train MLP, CNN and ViT on shapes in the left half; test in the left half and in the unseen right half.
4. Shift the object 4 pixels: does the CNN feature map shift with it (equivariance), and does each model's
   pooled representation stay put (invariance)?
"""

import _setup  # noqa: F401

import torch
import torch.nn.functional as F

from archlab.counts import n_params
from archlab.env import start
from archlab.vision import CNN, MLP, SHIFTED_COLS, TRAIN_COLS, TinyViT, receptive_field_width, shapes_batch

start("02_image_priors")

print("1. Dense weights against shared weights")
print(f"   224x224x3 image into a dense layer of width 1024: {224 * 224 * 3 * 1024 / 1e6:.1f} M weights")
print(f"   16x16 image into 128 hidden units: {MLP().body[1].weight.numel():,} weights; one 3x3 kernel: 9")
print()

print("2. Receptive field of L stacked 3x3 convolutions (stride 1), measured by gradient")
for L in range(1, 6):
    print(f"   L={L}: {receptive_field_width(L):>2} input columns   formula 1 + L(k-1) = {1 + 2 * L}")
print()

STEPS, BATCH, SEEDS = 1500, 64, [0, 1, 2]


def accuracy(model, cols, gen):
    x, y = shapes_batch(2000, cols, gen)
    with torch.no_grad():
        return (model(x).argmax(-1) == y).float().mean().item()


print(f"3. Train on columns {TRAIN_COLS}, test on the same columns and on unseen columns {SHIFTED_COLS}")
print(f"   {STEPS} steps, batch {BATCH}, Adam lr 1e-3, 4 classes (chance 0.25), seeds {SEEDS}")
print(f"   {'model':>6} {'params':>7} {'train cols':>11} {'shifted cols':>13}")
trained = {}
for name, cls in [("MLP", MLP), ("CNN", CNN), ("ViT", TinyViT)]:
    accs = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        gen = torch.Generator().manual_seed(seed)
        model = cls()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        for _ in range(STEPS):
            x, y = shapes_batch(BATCH, TRAIN_COLS, gen)
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        eval_gen = torch.Generator().manual_seed(123)
        accs.append((accuracy(model, TRAIN_COLS, eval_gen), accuracy(model, SHIFTED_COLS, eval_gen)))
        if seed == SEEDS[0]:
            trained[name] = model
    a = torch.tensor(accs)
    rng = lambda c: f"{a[:, c].mean():.3f} [{a[:, c].min():.2f}-{a[:, c].max():.2f}]"  # noqa: E731
    print(f"   {name:>6} {n_params(model):>7} {rng(0):>11} {rng(1):>13}")
print("   (mean over seeds, [min-max])")
print()

print("4. Move the object 4 pixels right (columns 2-5 -> 6-9), same noise, same row")
gen = torch.Generator().manual_seed(7)
x, _ = shapes_batch(500, (2, 5), gen, noise=0.0)
x = x + 0.1 * torch.randn(x.shape, generator=gen)
x_shift = torch.roll(x, 4, dims=3)  # nothing wraps: the shape ends by column 9 + 4 = 13 < 16
with torch.no_grad():
    for name, model in trained.items():
        f0, f1 = model.features(x), model.features(x_shift)
        rel = ((f1 - f0).norm(dim=1) / f0.norm(dim=1)).mean().item()
        same = (model(x).argmax(-1) == model(x_shift).argmax(-1)).float().mean().item()
        print(f"   {name}: relative change of pooled features {rel:.3f}; same prediction after shift {same:.3f}")
    fm0, fm1 = trained["CNN"].feature_map(x), trained["CNN"].feature_map(x_shift)
    inner = (slice(None), slice(None), slice(3, -3), slice(7, -3))  # away from the zero-padded borders
    err = (fm1[inner] - torch.roll(fm0, 4, dims=3)[inner]).abs().max().item()
    print(f"   CNN feature map: max |f(shift x) - shift f(x)| inside the borders = {err:.2e} "
          f"(feature scale {fm0.abs().max().item():.2f})")
    circ = CNN(padding_mode="circular")
    err_c = (circ.feature_map(x_shift) - torch.roll(circ.feature_map(x), 4, dims=3)).abs().max().item()
    print(f"   Same CNN shape with circular padding, whole map: max error {err_c:.2e}")
