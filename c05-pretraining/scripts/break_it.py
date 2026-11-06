"""Break training on purpose, watch each guard fire, and recover.

    python c05-pretraining/scripts/break_it.py --case all
    python c05-pretraining/scripts/break_it.py --case oom --device cuda
"""

from __future__ import annotations

import argparse
import resource
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from pretrain.checkpoint import latest_valid_checkpoint, load_checkpoint, verify  # noqa: E402
from pretrain.config import TrainConfig  # noqa: E402
from pretrain.guards import (BadBatchError, NonFiniteGuard, SpikeDetector, TrainingDiverged,  # noqa: E402
                             check_batch, step_with_oom_retry)
from pretrain.loader import PackedWindows, ResumableLoader  # noqa: E402
from pretrain.model import build_model  # noqa: E402
from pretrain.optim import build_optimizer  # noqa: E402
from pretrain.schedule import build_scheduler  # noqa: E402
from pretrain.shards import TokenShards  # noqa: E402
from pretrain.step import train_step  # noqa: E402
from pretrain.trainer import empty_cache, train  # noqa: E402
from pretrain.utils import pick_device, resolve, seed_everything  # noqa: E402

CFG = resolve("configs/smoke.yaml")


def setup(device: torch.device, lr: float = 3e-3, warmup: int = 10, total: int = 200):
    cfg = TrainConfig.from_yaml(CFG, lr=lr, warmup_steps=warmup, max_steps=total, device=str(device))
    seed_everything(0)
    model = build_model(cfg.model).to(device)
    opt = build_optimizer(model, cfg, device)
    sched = build_scheduler(opt, cfg.warmup_steps, cfg.max_steps, cfg.lr, min(cfg.min_lr, cfg.lr))
    loader = ResumableLoader(PackedWindows(TokenShards(resolve(cfg.data_dir), "train"), cfg.seq_len),
                             cfg.micro_batch_size, cfg.seed)
    return cfg, model, opt, sched, loader


def too_high_lr(device: torch.device) -> None:
    """The same 150 steps at a held LR, so no decay can rescue a bad choice. Only LR and clipping change."""
    runs = {"lr=3e-3": (3e-3, 1.0), "lr=3e-1": (3e-1, 1.0), "lr=3e+0": (3.0, 1.0), "lr=3e+0 no clip": (3.0, float("inf"))}
    print("step" + "".join(f"{name:>17}" for name in runs))
    rows = []
    for lr, clip in runs.values():
        cfg, model, opt, _, loader = setup(device, lr, 1, 150)
        losses = []
        for _ in range(150):
            micro = [next(loader) for _ in range(cfg.grad_accum_steps)]
            losses.append(train_step(model, opt, None, micro, device, "fp32", clip).loss)  # no scheduler: LR is held
        rows.append(losses)
    for s in (0, 1, 2, 4, 9, 19, 49, 99, 149):
        print(f"{s + 1:4d}" + "".join(f"{r[s]:17.3f}" for r in rows))


def nan_gradient(device: torch.device) -> None:
    """A forward hook poisons chosen steps. One bad step is skipped; three in a row escalate."""
    cfg, model, opt, sched, loader = setup(device)
    guard, poison = NonFiniteGuard(patience=3), {"on": False}

    def hook(_m, _in, out):
        return out * float("nan") if poison["on"] else out

    model.lm_head.register_forward_hook(hook)
    step = 0
    try:
        while True:
            step += 1
            poison["on"] = step == 5 or step >= 10
            micro = [next(loader) for _ in range(cfg.grad_accum_steps)]
            before = model.lm_head.weight.detach().clone()
            r = train_step(model, opt, sched, micro, device, "fp32", cfg.grad_clip)
            moved = not torch.equal(before, model.lm_head.weight)
            print(f"step {step:2d} loss {r.loss:8.4f} grad_norm {r.grad_norm:8.4f} "
                  f"skipped {r.skipped!s:5} weights moved {moved}")
            guard.update(r.skipped)
    except TrainingDiverged as e:
        print(f"TrainingDiverged: {e}. The loop would now reload the last good checkpoint and lower the LR.")


def bad_batch(device: torch.device) -> None:
    """A token id past the vocabulary, with and without the guard."""
    cfg, model, *_ = setup(device)
    x = torch.randint(0, cfg.model.vocab_size, (2, 16))
    y = torch.roll(x, -1, 1)
    x[0, 3] = cfg.model.vocab_size + 904  # e.g. a shard written with another tokenizer
    try:
        check_batch(x, y, cfg.model.vocab_size)
    except BadBatchError as e:
        print(f"with the guard: BadBatchError: {e}")
    try:
        logits = model(x.to(device)).logits
        print(f"without the guard on {device}: no error, loss {logits.float().mean().item():.4f} "
              f"finite {bool(torch.isfinite(logits).all())}")
    except Exception as e:  # noqa: BLE001
        print(f"without the guard on {device}: {type(e).__name__}: {str(e)[:90]}")


def spike(device: torch.device) -> None:
    """Train, then feed a batch of uniform random tokens: real loss jump, caught by the detector."""
    cfg, model, opt, sched, loader = setup(device, total=120)
    det = SpikeDetector()
    for s in range(1, 111):
        micro = [next(loader) for _ in range(cfg.grad_accum_steps)]
        if s == 100:
            micro = [(torch.randint(0, cfg.model.vocab_size, x.shape),) * 2 for x, _ in micro]
            micro = [(x, torch.roll(x, -1, 1)) for x, _ in micro]
        r = train_step(model, opt, sched, micro, device, "fp32", cfg.grad_clip)
        flagged = det.update(r.loss)
        if s in (98, 99, 100, 101, 102) or flagged:
            print(f"step {s:3d} loss {r.loss:.3f} recent median {det.median:.3f} spread {det.spread:.3f} "
                  f"grad_norm {r.grad_norm:.3f} spike {flagged}")


def cap_memory(device: torch.device, limit_mb: int):
    """Limit this process only, so the failure is real and no neighbour is affected.

    Returns a function that lifts the limit.
    """
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(limit_mb * 2**20 / torch.cuda.get_device_properties(device).total_memory)
        return lambda: torch.cuda.set_per_process_memory_fraction(1.0)
    if device.type == "mps":
        torch.mps.set_per_process_memory_fraction(limit_mb * 2**20 / torch.mps.recommended_max_memory())
        return lambda: torch.mps.set_per_process_memory_fraction(0.0)
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)  # address space, enforced on Linux
    used = int(Path("/proc/self/statm").read_text().split()[0]) * resource.getpagesize()
    resource.setrlimit(resource.RLIMIT_AS, (used + limit_mb * 2**20, hard))
    return lambda: resource.setrlimit(resource.RLIMIT_AS, (soft, hard))


def oom(device: torch.device) -> None:
    """A real out-of-memory, forced by capping the process, then recovered by splitting."""
    cfg = TrainConfig.from_yaml(resolve("configs/tinystories.yaml"), device=str(device))
    precision = "fp32" if device.type == "cpu" else "bf16"
    seed_everything(0)
    model = build_model(cfg.model).to(device)
    opt = build_optimizer(model, cfg, device)
    x = torch.randint(0, cfg.model.vocab_size, (128, 256))
    micro = [(x, torch.roll(x, -1, 1))]
    limit_mb = 1024
    print(f"memory cap {limit_mb} MiB above current use, one micro-batch of {x.size(0)}")
    fn = lambda mb: train_step(model, opt, None, mb, device, precision, 1.0)  # noqa: E731
    release = cap_memory(device, limit_mb)
    try:
        res, n = step_with_oom_retry(fn, micro, opt, lambda: empty_cache(device), max_splits=5)
    finally:
        release()
    print(f"finished with {n} micro-batches, loss {res.loss:.4f}, tokens {res.tokens:,} (all of them)")


def corrupt_checkpoint(device: torch.device) -> None:
    """Truncate the newest checkpoint, then resume."""
    out = "runs/break_corrupt"
    shutil.rmtree(resolve(out), ignore_errors=True)
    cfg = TrainConfig.from_yaml(CFG, out_dir=out, max_steps=20, ckpt_every=10, eval_every=10_000, log_every=100)
    train(cfg, resume=False)
    newest = sorted(resolve(out).glob("ckpt_*.pt"))[-1]
    data = newest.read_bytes()
    newest.write_bytes(data[: len(data) // 2])
    print(f"truncated {newest.name}; verify -> {verify(newest)}")
    try:
        torch.load(newest, weights_only=True)
    except Exception as e:  # noqa: BLE001
        print(f"plain torch.load: {type(e).__name__}: {str(e)[:80]}")
    good = latest_valid_checkpoint(resolve(out))
    print(f"latest valid: {good.name}")
    state = torch.load(good, weights_only=True)
    print(f"resumable from step {state['step']}, data position epoch {state['data']['epoch']} "
          f"batch {state['data']['consumed']}")
    train(cfg, resume=True)
    print(f"finished; newest checkpoint is {sorted(resolve(out).glob('ckpt_*.pt'))[-1].name}")
    shutil.rmtree(resolve(out), ignore_errors=True)


CASES = {"too_high_lr": too_high_lr, "nan_gradient": nan_gradient, "bad_batch": bad_batch,
         "spike": spike, "oom": oom, "corrupt_checkpoint": corrupt_checkpoint}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="all", choices=["all", *CASES])
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    device = pick_device(args.device)
    for name in CASES if args.case == "all" else [args.case]:
        print(f"\n=== {name} ({device}) ===")
        CASES[name](device)


if __name__ == "__main__":
    main()
