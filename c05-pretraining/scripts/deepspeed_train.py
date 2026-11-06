"""DeepSpeed ZeRO stages 1 to 3 on the real data. GPU only, needs `pip install deepspeed`.

    torchrun --nproc_per_node 4 c05-pretraining/scripts/deepspeed_train.py --stage 3 --steps 50

Run it once per stage and compare the peak memory lines. The engine does what
pretrain/step.py does by hand: accumulation, clipping, bf16 and the optimizer step.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import deepspeed  # noqa: E402
import torch  # noqa: E402

from pretrain.config import TrainConfig  # noqa: E402
from pretrain.distributed import cleanup, init_distributed  # noqa: E402
from pretrain.loader import PackedWindows, ResumableLoader  # noqa: E402
from pretrain.model import build_model  # noqa: E402
from pretrain.shards import TokenShards  # noqa: E402
from pretrain.step import lm_loss_sum  # noqa: E402
from pretrain.utils import ROOT, resolve, seed_everything  # noqa: E402
from pretrain.zero import deepspeed_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/tinystories.yaml"))
    ap.add_argument("--stage", type=int, default=3, choices=[1, 2, 3])
    ap.add_argument("--steps", type=int, default=50)
    args = ap.parse_args()
    info = init_distributed()
    device = torch.device(f"cuda:{info.local_rank}")
    cfg = TrainConfig.from_yaml(args.config)
    seed_everything(cfg.seed)

    ds_config = deepspeed_config(args.stage, cfg.micro_batch_size, cfg.grad_accum_steps, cfg.lr, cfg.grad_clip)
    engine, *_ = deepspeed.initialize(model=build_model(cfg.model), config=ds_config)
    loader = ResumableLoader(PackedWindows(TokenShards(resolve(cfg.data_dir), "train"), cfg.seq_len),
                             cfg.micro_batch_size, cfg.seed, info.rank, info.world_size, pin_memory=True)

    torch.cuda.reset_peak_memory_stats()
    t0, micro_steps = time.time(), args.steps * cfg.grad_accum_steps
    for i in range(micro_steps):
        x, y = next(loader)
        x, y = x.to(device), y.to(device)
        logits = engine(x).logits  # (B, L, V). Parameters are bf16 inside the engine.
        loss = lm_loss_sum(logits, y) / y.numel()
        engine.backward(loss)  # divides by gradient_accumulation_steps itself
        engine.step()  # applies the optimizer only on every N-th call
        if info.is_main and (i + 1) % (10 * cfg.grad_accum_steps) == 0:
            print(f"step {(i + 1) // cfg.grad_accum_steps} loss {loss.item():.4f}", flush=True)

    torch.cuda.synchronize()
    tok_s = micro_steps * cfg.micro_batch_size * cfg.seq_len * info.world_size / (time.time() - t0)
    print(f"rank {info.rank}: ZeRO stage {args.stage}, {info.world_size} GPUs, {tok_s:,.0f} tokens/s in total, "
          f"peak {torch.cuda.max_memory_allocated() / 2**20:.1f} MiB", flush=True)
    cleanup()


if __name__ == "__main__":
    main()
