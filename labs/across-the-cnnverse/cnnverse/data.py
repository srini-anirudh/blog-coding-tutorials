"""CIFAR-10 as uint8 tensors, loaded once from the Hugging Face Hub and cached next to the HF cache."""

import os
from pathlib import Path

import numpy as np
import torch

MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)


def cache_path():
    root = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    return root / "cnnverse_cifar10_uint8.pt"


def load_cifar10():
    """Returns {'train': (x uint8 N,3,32,32, y int64), 'test': (...)}. Downloads about 120 MB the first time."""
    path = cache_path()
    if path.exists():
        return torch.load(path)
    from datasets import load_dataset
    ds = load_dataset("uoft-cs/cifar10")
    out = {}
    for split in ("train", "test"):
        imgs = np.stack([np.asarray(im, dtype=np.uint8) for im in ds[split]["img"]])
        out[split] = (torch.from_numpy(imgs).permute(0, 3, 1, 2).contiguous(), torch.tensor(ds[split]["label"]))
    torch.save(out, path)
    return out


def synthetic(n=256, seed=0):
    """A stand-in with CIFAR's shapes, for tests: the class is which quadrant holds a bright square."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, 60, (n, 3, 32, 32), generator=g, dtype=torch.uint8)
    y = torch.randint(0, 4, (n,), generator=g)
    for i in range(n):
        r, c = divmod(int(y[i]), 2)
        x[i, :, 16 * r + 4:16 * r + 12, 16 * c + 4:16 * c + 12] = 255
    return x, y


def normalise(x_uint8):
    return (x_uint8.float() / 255 - MEAN.to(x_uint8.device)) / STD.to(x_uint8.device)


def augment(x):
    """Pad 4 and random-crop 32, plus random horizontal flip: the augmentation of He et al. 2015, on tensors."""
    n = x.shape[0]
    pad = torch.nn.functional.pad(x, (4, 4, 4, 4))
    i, j = torch.randint(0, 9, (2,), device=x.device).tolist()   # one offset per batch keeps it cheap
    x = pad[:, :, i:i + 32, j:j + 32]
    flip = torch.rand(n, device=x.device) < 0.5
    return torch.where(flip.view(-1, 1, 1, 1), x.flip(-1), x)
