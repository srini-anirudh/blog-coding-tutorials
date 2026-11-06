"""Section 04, failure mode 1: small batches. A small CNN on a CIFAR-10 subset with BatchNorm, GroupNorm or no norm.

Every run sees the same images for the same number of epochs. The learning rate follows the linear scaling rule
(lr = base_lr * batch / 64), so smaller batches take more, smaller steps.

For BatchNorm we report two test accuracies:
  eval mode         running EMA statistics, the function you would deploy
  test-batch stats  statistics of each test batch of 256, which shows how much of the loss is the running stats
This is a toy: 10k training images, 2 epochs. It shows the direction of the effect, not ImageNet numbers.
"""

import argparse
import io
import time

import _setup  # noqa: F401
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from PIL import Image
from torch import nn

from normgeo.report import Report

p = argparse.ArgumentParser()
p.add_argument("--train", type=int, default=10000)
p.add_argument("--test", type=int, default=2000)
p.add_argument("--epochs", type=int, default=2)
p.add_argument("--batches", type=int, nargs="+", default=[2, 8, 64])
p.add_argument("--base-lr", type=float, default=0.05)
args = p.parse_args()


def load(split: str, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    path = hf_hub_download("uoft-cs/cifar10", f"plain_text/{split}-00000-of-00001.parquet", repo_type="dataset")
    df = pd.read_parquet(path).iloc[:n]
    imgs = np.stack([np.asarray(Image.open(io.BytesIO(b["bytes"])).convert("RGB")) for b in df["img"]])
    x = torch.tensor(imgs).permute(0, 3, 1, 2).float() / 255
    mean, std = torch.tensor([0.491, 0.482, 0.447]), torch.tensor([0.247, 0.243, 0.262])
    return (x - mean[:, None, None]) / std[:, None, None], torch.tensor(df["label"].values)


def net(norm: str) -> nn.Sequential:
    def nl(c: int) -> nn.Module:
        return {"batchnorm": nn.BatchNorm2d(c), "groupnorm": nn.GroupNorm(8, c), "none": nn.Identity()}[norm]

    layers, c_in = [], 3
    for c in (32, 64, 128):
        layers += [nn.Conv2d(c_in, c, 3, padding=1, bias=norm == "none"), nl(c), nn.ReLU(), nn.MaxPool2d(2)]
        c_in = c
    return nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, 10))


@torch.no_grad()
def accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor, train_mode: bool = False) -> float:
    model.train(train_mode)
    correct = sum((model(x[i : i + 256]).argmax(-1) == y[i : i + 256]).sum().item() for i in range(0, len(x), 256))
    return correct / len(x)


torch.manual_seed(0)
xtr, ytr = load("train", args.train)
xte, yte = load("test", args.test)

with Report("05_small_batch") as say:
    say(f"CIFAR-10, {args.train} train / {args.test} test images, {args.epochs} epochs, SGD momentum 0.9, "
        f"lr = {args.base_lr} * batch / 64, random flips.")
    say(f"{'norm':<11}{'batch':>6}{'steps':>7}{'train loss':>12}{'test acc (eval)':>17}{'test-batch stats':>18}{'time s':>8}")
    rows = []
    for norm in ["batchnorm", "groupnorm", "none"]:
        for B in args.batches:
            torch.manual_seed(1)
            model = net(norm)
            opt = torch.optim.SGD(model.parameters(), lr=args.base_lr * B / 64, momentum=0.9)
            t0, steps, losses = time.time(), 0, []
            for _ in range(args.epochs):
                model.train()
                perm = torch.randperm(len(xtr))
                for i in range(0, len(xtr) - B + 1, B):
                    idx = perm[i : i + B]
                    xb = xtr[idx]
                    xb = torch.where(torch.rand(B, 1, 1, 1) < 0.5, xb.flip(-1), xb)
                    loss = F.cross_entropy(model(xb), ytr[idx])
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    steps += 1
                    losses.append(loss.item())
            k = max(1, len(losses) // 10)
            tail = sum(losses[-k:]) / k
            acc = accuracy(model, xte, yte)
            acc_bs = accuracy(model, xte, yte, train_mode=True) if norm == "batchnorm" else float("nan")
            rows.append({"norm": norm, "batch": B, "steps": steps, "train_loss": tail, "test_acc": acc, "test_acc_batch_stats": acc_bs})
            bs_txt = f"{acc_bs:.3f}" if norm == "batchnorm" else "-"
            say(f"{norm:<11}{B:>6}{steps:>7}{tail:>12.3f}{acc:>17.3f}{bs_txt:>18}{time.time() - t0:>8.0f}")
    say("train loss = mean over the last 10% of steps.")
    say.json({"args": vars(args), "rows": rows})
