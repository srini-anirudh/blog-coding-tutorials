"""Train the spiral MLP.

Run:
    python c01-pytorch/scripts/05_train.py --optimizer adamw --lr 1e-2
    python c01-pytorch/scripts/05_train.py --config c01-pytorch/configs/spiral_adamw.yaml
    python c01-pytorch/scripts/05_train.py --compare            # manual vs torch optimizers
    python c01-pytorch/scripts/05_train.py --export run.json    # history + decision boundaries
"""

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import yaml

from scratchnet.data import SpiralDataset
from scratchnet.train import TrainConfig, fit

LRS = {"manual-sgd": 0.1, "sgd": 0.1, "manual-adamw": 1e-2, "adamw": 1e-2}


def boundary(model: torch.nn.Module, n: int = 48, lim: float = 1.1) -> list[list[int]]:
    """Predicted class on an n x n grid over [-lim, lim]^2, row 0 at the top."""
    xs = torch.linspace(-lim, lim, n)
    ys = torch.linspace(lim, -lim, n)
    grid = torch.stack(torch.meshgrid(ys, xs, indexing="ij")[::-1], dim=-1).reshape(-1, 2)  # [n*n, (x, y)]
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        pred = model(grid.to(device)).argmax(-1).cpu()
    model.train()
    return pred.reshape(n, n).tolist()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=None, help="YAML file of TrainConfig fields")
    ap.add_argument("--optimizer", default="adamw", choices=list(LRS))
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--export", type=Path, default=None)
    args = ap.parse_args()

    if args.compare:
        print(f"{'optimizer':<13} {'lr':>6} {'train_loss':>10} {'val_loss':>9} {'val_acc':>8} {'time_s':>7}")
        histories = {}
        for name, lr in LRS.items():
            t0 = time.perf_counter()
            _, histories[name] = fit(TrainConfig(optimizer=name, lr=lr, epochs=args.epochs, device=args.device))
            last = histories[name][-1]
            print(f"{name:<13} {lr:>6g} {last['train_loss']:>10.4f} {last['val_loss']:>9.4f} "
                  f"{last['val_acc']:>8.3f} {time.perf_counter() - t0:>7.2f}")
        for ours, ref in [("manual-sgd", "sgd"), ("manual-adamw", "adamw")]:
            gap = max(abs(a["train_loss"] - b["train_loss"]) for a, b in zip(histories[ours], histories[ref]))
            print(f"max |train_loss({ours}) - train_loss({ref})| over {args.epochs} epochs: {gap:.2e}")
        if args.export:
            args.export.write_text(json.dumps(histories))
        return

    if args.config:
        cfg = TrainConfig(**yaml.safe_load(args.config.read_text()))
    else:
        cfg = TrainConfig(optimizer=args.optimizer, lr=args.lr or LRS[args.optimizer], epochs=args.epochs,
                          device=args.device)
    snapshots = {}
    keep = {0, 1, 2, 4, 8, 16, 32, cfg.epochs - 1}

    def on_epoch_end(epoch, model, row):
        if epoch % 10 == 0 or epoch == cfg.epochs - 1:
            print(f"epoch {epoch:>3}  train_loss {row['train_loss']:.4f}  val_loss {row['val_loss']:.4f}  "
                  f"val_acc {row['val_acc']:.3f}  max_grad_norm {row['grad_norm']:.3f}")
        if args.export and epoch in keep:
            snapshots[epoch] = boundary(model)

    t0 = time.perf_counter()
    _, history = fit(cfg, on_epoch_end=on_epoch_end)
    print(f"{cfg.epochs} epochs in {time.perf_counter() - t0:.2f} s on {cfg.device}")
    if args.export:
        ds = SpiralDataset(cfg.points_per_class, seed=cfg.seed)
        args.export.write_text(json.dumps({
            "config": asdict(cfg), "history": history, "boundaries": snapshots,
            "points": [[round(x, 3), round(y, 3), int(c)] for (x, y), c in zip(ds.x.tolist()[::3], ds.y.tolist()[::3])],
        }))
        print("wrote", args.export)


if __name__ == "__main__":
    main()
