"""Why does C4's TinyLLM end lower than HF Llama, when the two compute the same function?

The only large difference between them is the start. HF draws every weight with std 0.02.
C4 also divides the residual projections (o_proj, down_proj) by sqrt(2 * num_layers).
So we train HF Llama again from its own seeded weights with only those matrices scaled.

Run after both main runs, because it reads their metrics:

    python c05-pretraining/scripts/train.py --config c05-pretraining/configs/tinystories.yaml --precision fp32 --device cpu --out-dir runs/tinystories
    python c05-pretraining/scripts/train.py --config c05-pretraining/configs/tinystories.yaml --precision fp32 --device cpu --model tinyllm --out-dir runs/tinyllm
    python c05-pretraining/scripts/compare_init.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from pretrain import trainer  # noqa: E402
from pretrain.config import TrainConfig  # noqa: E402
from pretrain.metrics import read_metrics  # noqa: E402
from pretrain.utils import resolve  # noqa: E402

RUNS = {"llama, flat init": "runs/tinystories", "tinyllm": "runs/tinyllm", "llama, scaled init": "runs/llama_scaled_init"}


def scale_residual_projections(model: torch.nn.Module, num_layers: int) -> None:
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith(("o_proj.weight", "down_proj.weight")):
                p /= (2 * num_layers) ** 0.5


def main() -> None:
    cfg = TrainConfig.from_yaml(resolve("configs/tinystories.yaml"), precision="fp32", device="cpu",
                                out_dir=RUNS["llama, scaled init"], max_minutes=28)
    original = trainer.build_model

    def build_scaled(a, model_type):
        model = original(a, model_type)
        scale_residual_projections(model, a.num_layers)
        return model

    trainer.build_model = build_scaled
    try:
        trainer.train(cfg, resume=False)
    finally:
        trainer.build_model = original

    curves = {name: {r["step"]: r["val_loss"] for r in read_metrics(resolve(path) / "metrics.jsonl") if "val_loss" in r}
              for name, path in RUNS.items()}
    print(f"\n{'step':>5}" + "".join(f"{name:>20}" for name in curves))
    for step in (50, 100, 200, 400, 600, 800):
        print(f"{step:>5}" + "".join(f"{c[step]:>20.4f}" for c in curves.values()))


if __name__ == "__main__":
    main()
