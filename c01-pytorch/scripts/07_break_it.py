"""Break the training loop on purpose, then find the bug with hooks and checks.

Run: python c01-pytorch/scripts/07_break_it.py [lr|zero_grad|eval|shape|device]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader

from scratchnet.data import SpiralDataset, collate_dicts
from scratchnet.debug import GradNormMonitor, NaNDetector, check_same_device
from scratchnet.device import pick_device
from scratchnet.layers import MLP
from scratchnet.losses import cross_entropy
from scratchnet.train import TrainConfig, evaluate, fit


def batches(ds: SpiralDataset, batch_size: int = 64, seed: int = 0):
    perm = torch.randperm(len(ds), generator=torch.Generator().manual_seed(seed))
    for i in range(0, len(ds), batch_size):
        idx = perm[i : i + batch_size]
        yield ds.x[idx], ds.y[idx]


def break_lr() -> None:
    print("== learning rate 10 with plain SGD ==")
    torch.manual_seed(0)
    ds = SpiralDataset()
    model = MLP(2, 64, 3, mean=ds.x.mean(0), std=ds.x.std(0))
    opt = torch.optim.SGD(model.parameters(), lr=10.0, momentum=0.9)
    step = 0
    with NaNDetector(model), GradNormMonitor(model) as mon:
        try:
            for epoch in range(3):
                for x, y in batches(ds, seed=epoch):
                    opt.zero_grad()
                    loss = cross_entropy(model(x), y)
                    loss.backward()
                    worst = max(mon.record.norms, key=lambda n: mon.record.norms[n])
                    if step % 3 == 0:
                        print(f"step {step:>2}: loss {loss.item():>10.4g}  largest grad norm "
                              f"{mon.record.norms[worst].item():>9.3g} ({worst})")
                    opt.step()
                    mon.reset()
                    step += 1
        except FloatingPointError as err:
            print(f"step {step:>2}: NaNDetector: {err}")
    print("weight norm of blocks.0.0:", f"{model.blocks[0][0].weight.norm().item():.3g}")
    for name, kwargs in [("lr 10, clip 1.0", dict(lr=10.0, max_grad_norm=1.0)), ("lr 0.1", dict(lr=0.1))]:
        _, hist = fit(TrainConfig(optimizer="sgd", epochs=30, **kwargs))
        print(f"try: {name:<16} -> final train_loss {hist[-1]['train_loss']:.4f}  val_acc {hist[-1]['val_acc']:.3f}")


def break_zero_grad() -> None:
    print("== forgetting optimizer.zero_grad() ==")
    for forget in (False, True):
        torch.manual_seed(0)
        ds = SpiralDataset()
        model = MLP(2, 64, 3, mean=ds.x.mean(0), std=ds.x.std(0))
        opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        for epoch in range(30):
            for x, y in batches(ds, seed=epoch):
                if not forget:
                    opt.zero_grad()
                cross_entropy(model(x), y).backward()
                opt.step()
        g = model.head.weight.grad.norm().item()
        print(f"zero_grad {'skipped' if forget else 'called '}: loss {cross_entropy(model(ds.x), ds.y).item():.4f}  "
              f"head.weight.grad norm after 30 epochs {g:.3g}")


def break_eval() -> None:
    print("== evaluating without model.eval() (dropout 0.3) ==")
    model, _ = fit(TrainConfig(epochs=30, dropout=0.3))
    val_ds = SpiralDataset(seed=1)
    model.train()  # the mistake: dropout stays on
    with torch.no_grad():
        a, b = model(val_ds.x).argmax(-1), model(val_ds.x).argmax(-1)
    print(f"train mode: two forward passes disagree on {(a != b).float().mean().item():.1%} of points")
    print(f"accuracy in train mode {(a == val_ds.y).float().mean().item():.3f}")
    val = DataLoader(val_ds, batch_size=512, collate_fn=collate_dicts)
    print(f"accuracy in eval mode  {evaluate(model, val, torch.device('cpu'))['val_acc']:.3f}")


def break_shape() -> None:
    print("== wrong input width ==")
    model = MLP(2, 16, 3)
    try:
        model.blocks[0][0](torch.randn(8, 3))
    except RuntimeError as err:
        print("RuntimeError:", err)


def break_device() -> None:
    print("== model and batch on different devices ==")
    # The meta device holds shapes but no data, so a CPU-only machine still hits the real check.
    device = pick_device().type if pick_device().type != "cpu" else "meta"
    model = MLP(2, 16, 3).to(device)
    x = torch.randn(8, 2)
    try:
        model(x)
    except RuntimeError as err:
        print("RuntimeError:", str(err).splitlines()[0])
    try:
        check_same_device(model, x=x)
    except RuntimeError as err:
        print("check_same_device:", err)


SCENARIOS = {"lr": break_lr, "zero_grad": break_zero_grad, "eval": break_eval, "shape": break_shape,
             "device": break_device}

if __name__ == "__main__":
    for name in sys.argv[1:] or SCENARIOS:
        SCENARIOS[name]()
        print()
