"""Print and plot the warmup + cosine schedule the main run uses.

    python c05-pretraining/scripts/plot_schedule.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pretrain.config import TrainConfig  # noqa: E402
from pretrain.schedule import lr_multiplier  # noqa: E402
from pretrain.utils import resolve  # noqa: E402


def main() -> None:
    c = TrainConfig.from_yaml(resolve("configs/tinystories.yaml"))
    ratio = c.min_lr / c.lr
    lrs = [c.lr * lr_multiplier(s, c.warmup_steps, c.max_steps, ratio) for s in range(c.max_steps)]
    for s in (0, 1, c.warmup_steps // 2, c.warmup_steps - 1, c.warmup_steps, c.max_steps // 2, c.max_steps - 1):
        print(f"step {s:4d}  lr {lrs[s]:.3e}")
    out = resolve("runs/lr_schedule.png")
    out.parent.mkdir(exist_ok=True)
    plt.figure(figsize=(7, 3))
    plt.plot(lrs)
    plt.xlabel("optimiser step")
    plt.ylabel("learning rate")
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    print(f"wrote {out.name}")


if __name__ == "__main__":
    main()
