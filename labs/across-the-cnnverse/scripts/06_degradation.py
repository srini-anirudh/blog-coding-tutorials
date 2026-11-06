"""The degradation problem that motivated ResNet, at small scale on CIFAR-10.

Trains plain and residual CIFAR networks (He et al. 2015, section 4.2) of depth 20 and 56. Plain and residual
nets have identical weights: the option-A shortcut has no parameters. The claim to check: the deeper plain net
has higher *training* error than the shallower one, and the residual nets do not degrade.

Short schedule: the paper trains 64k iterations (about 164 epochs); this runs --epochs (default 30) with the
learning rate divided by 10 at 50% and 75% of training. The README runs use --epochs 20.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from cnnverse import data, zoo
from cnnverse.report import RESULTS, Report, set_threads


def evaluate(model, x, y, bs=1000):
    model.eval()
    wrong = 0
    with torch.no_grad(), torch.autocast(x.device.type, torch.bfloat16, enabled=x.device.type == "cuda"):
        for i in range(0, len(x), bs):
            wrong += (model(data.normalise(x[i:i + bs])).argmax(1) != y[i:i + bs]).sum().item()
    return wrong / len(x)


def train(depth, residual, xtr, ytr, xte, yte, epochs, bs=128, lr=0.1, seed=0, classes=10, log=print):
    torch.manual_seed(seed)
    dev = xtr.device
    model = zoo.cifar_net(depth, residual, classes=classes).to(dev).to(memory_format=torch.channels_last)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    steps = epochs * (len(xtr) // bs)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, [steps // 2, steps * 3 // 4], 0.1)
    hist = []
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(xtr), device=dev)
        wrong = seen = 0
        loss_sum = 0.0
        for i in range(0, len(xtr) - bs + 1, bs):
            idx = perm[i:i + bs]
            xb, yb = data.normalise(data.augment(xtr[idx])), ytr[idx]
            with torch.autocast(dev.type, torch.bfloat16, enabled=dev.type == "cuda"):
                out = model(xb.contiguous(memory_format=torch.channels_last))
                loss = F.cross_entropy(out, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            wrong += (out.argmax(1) != yb).sum().item()
            seen += len(yb)
            loss_sum += loss.item() * len(yb)
        hist.append({"epoch": ep + 1, "train_err": wrong / seen, "train_loss": loss_sum / seen})
        log(f"  depth {depth:>3} {'residual' if residual else 'plain':>8} epoch {ep + 1:>3}"
            f"  train err {wrong / seen:.4f}  loss {loss_sum / seen:.4f}  {time.time() - t0:6.1f}s")
    clean_train = evaluate(model, xtr[:10000], ytr[:10000])
    test = evaluate(model, xte, yte)
    return {"depth": depth, "residual": residual, "params": sum(p.numel() for p in model.parameters()),
            "history": hist, "final_train_err_augmented": hist[-1]["train_err"],
            "train_err_clean_10k": clean_train, "test_err": test, "seconds": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", type=int, nargs="+", default=[20, 56])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--subset", type=int, default=0, help="use only this many training images (0 = all 50k)")
    ap.add_argument("--synthetic", action="store_true", help="tiny synthetic data, for a smoke test")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    set_threads()
    dev = torch.device(args.device)
    name = f"06_degradation{args.tag}"
    R = Report(f"{name}.txt", device=args.device)
    if args.synthetic:
        (xtr, ytr), (xte, yte), classes = data.synthetic(512), data.synthetic(256, seed=1), 4
    else:
        d = data.load_cifar10()
        (xtr, ytr), (xte, yte), classes = d["train"], d["test"], 10
    if args.subset:
        xtr, ytr = xtr[:args.subset], ytr[:args.subset]
    xtr, ytr, xte, yte = xtr.to(dev), ytr.to(dev), xte.to(dev), yte.to(dev)
    R(f"# data: {'synthetic' if args.synthetic else 'CIFAR-10'}, {len(xtr)} train / {len(xte)} test images,"
      f" {args.epochs} epochs, batch 128, SGD 0.1 momentum 0.9 wd 1e-4, lr /10 at 50% and 75%")
    runs = []
    for depth in args.depths:
        for residual in (False, True):
            runs.append(train(depth, residual, xtr, ytr, xte, yte, args.epochs, classes=classes, log=R))
    R(f"\n{'net':>14} {'params':>8} {'train err (aug, last epoch)':>28} {'train err (clean, 10k)':>23}"
      f" {'test err':>9} {'time':>7}")
    for r in runs:
        R(f"{('ResNet-' if r['residual'] else 'plain-') + str(r['depth']):>14} {r['params']:>8,}"
          f" {r['final_train_err_augmented']:>28.4f} {r['train_err_clean_10k']:>23.4f} {r['test_err']:>9.4f}"
          f" {r['seconds']:>6.0f}s")
    (RESULTS / f"{name}.json").write_text(json.dumps(runs, indent=1))
    R.close()


if __name__ == "__main__":
    main()
