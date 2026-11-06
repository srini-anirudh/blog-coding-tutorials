"""Our loop against Hugging Face Trainer: same weights, batches, optimizer and schedule.

If the loop is right, both end with the same weights. The Trainer only gets three things
it needs to match us: our loss, our optimizer pair, and a sequential sampler.

    python c05-pretraining/scripts/compare_hf_trainer.py --steps 20
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from torch.utils.data import DataLoader, Dataset, SequentialSampler  # noqa: E402
from transformers import Trainer, TrainingArguments  # noqa: E402

from pretrain.config import ModelArgs, TrainConfig  # noqa: E402
from pretrain.loader import PackedWindows  # noqa: E402
from pretrain.model import build_model  # noqa: E402
from pretrain.optim import build_optimizer  # noqa: E402
from pretrain.schedule import build_scheduler  # noqa: E402
from pretrain.shards import TokenShards  # noqa: E402
from pretrain.step import lm_loss_sum, train_step  # noqa: E402
from pretrain.utils import resolve  # noqa: E402

CPU = torch.device("cpu")


class AsDicts(Dataset):
    """The Trainer's collator wants one dict per sample."""

    def __init__(self, windows: PackedWindows):
        self.windows = windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, i: int) -> dict:
        x, y = self.windows[i]
        return {"input_ids": x, "targets": y}


class SameLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        logits = model(inputs["input_ids"]).logits  # (B, L, V)
        loss = lm_loss_sum(logits, inputs["targets"]) / inputs["targets"].numel()
        return (loss, logits) if return_outputs else loss

    def _get_train_sampler(self, *args, **kwargs):
        return SequentialSampler(self.train_dataset)


def fresh_parts(cfg: TrainConfig):
    torch.manual_seed(cfg.seed)
    model = build_model(cfg.model)
    opt = build_optimizer(model, cfg, CPU)
    return model, opt, build_scheduler(opt, cfg.warmup_steps, cfg.max_steps, cfg.lr, cfg.min_lr)


def ours(cfg: TrainConfig, windows: PackedWindows) -> tuple[list[float], torch.Tensor]:
    model, opt, sched = fresh_parts(cfg)
    batches = iter(DataLoader(windows, cfg.micro_batch_size, shuffle=False))
    losses = []
    for _ in range(cfg.max_steps):
        micro = [next(batches) for _ in range(cfg.grad_accum_steps)]
        losses.append(train_step(model, opt, sched, micro, CPU, "fp32", cfg.grad_clip).loss)
    return losses, torch.cat([p.detach().flatten() for p in model.parameters()])


def hf_trainer(cfg: TrainConfig, windows: PackedWindows) -> tuple[list[float], torch.Tensor]:
    model, opt, sched = fresh_parts(cfg)
    with tempfile.TemporaryDirectory() as tmp:
        args = TrainingArguments(
            output_dir=tmp, max_steps=cfg.max_steps, per_device_train_batch_size=cfg.micro_batch_size,
            gradient_accumulation_steps=cfg.grad_accum_steps, max_grad_norm=cfg.grad_clip, use_cpu=True,
            remove_unused_columns=False, report_to=[], save_strategy="no", logging_steps=1, disable_tqdm=True,
        )
        trainer = SameLossTrainer(model=model, args=args, train_dataset=AsDicts(windows), optimizers=(opt, sched))
        trainer.train()
    losses = [e["loss"] for e in trainer.state.log_history if "loss" in e]
    return losses, torch.cat([p.detach().flatten() for p in model.parameters()])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data_cache/tinystories")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--out", help="write the result as JSON (used by the test)")
    args = ap.parse_args()
    cfg = TrainConfig(
        seq_len=64, micro_batch_size=8, grad_accum_steps=2, max_steps=args.steps, warmup_steps=5,
        lr=3e-3, min_lr=3e-4, grad_clip=1.0,
        model=ModelArgs(hidden_size=64, num_layers=2, num_heads=4, num_kv_heads=2, intermediate_size=172, max_seq_len=64),
    )
    shards = TokenShards(resolve(args.data_dir), "train")
    cfg.model.vocab_size = shards.vocab_size
    windows = PackedWindows(shards, cfg.seq_len)

    loss_ours, w_ours = ours(cfg, windows)
    loss_hf, w_hf = hf_trainer(cfg, windows)
    gap = (w_ours - w_hf).abs().max().item()
    print(f"{args.steps} steps, 2 micro-batches of 8 per step, seq_len 64, fp32, same init and same batches")
    print(f"{'step':>5} {'our loop':>10} {'HF Trainer':>11}")
    for s in sorted({1, 2, 5, 10, args.steps} & set(range(1, args.steps + 1))):
        print(f"{s:>5} {loss_ours[s - 1]:>10.4f} {loss_hf[s - 1]:>11.4f}")
    print(f"max |weights ours - weights HF| after the last step: {gap:.2e}")
    if args.out:
        Path(args.out).write_text(json.dumps({"weight_gap": gap, "ours": loss_ours, "hf": loss_hf}))


if __name__ == "__main__":
    main()
