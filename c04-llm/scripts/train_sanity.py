"""A few minutes of training on Tiny Shakespeare: does the loss go down, and do samples get better?

This is a sanity check of the architecture, not a training recipe. C5 builds the real trainer
(packing, schedules, accumulation, mixed precision, checkpoint/resume).

    python c04-llm/scripts/train_sanity.py               # tiny preset, MPS if available
    python c04-llm/scripts/train_sanity.py --steps 300 --device cpu
"""

from __future__ import annotations

import argparse
import math
import sys
import time
import urllib.request
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tinyllm import ByteTokenizer, TinyLLM, get_config, save_checkpoint  # noqa: E402
from tinyllm.utils import pick_device, set_seed  # noqa: E402

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def load_corpus(path: Path) -> str:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading Tiny Shakespeare -> {path}")
        urllib.request.urlretrieve(DATA_URL, path)
    return path.read_text()


def random_batch(data: torch.Tensor, batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
    """(B, seq_len) windows at random offsets. labels = inputs; the model shifts internally."""
    starts = torch.randint(0, data.numel() - seq_len, (batch_size,))
    return torch.stack([data[s : s + seq_len] for s in starts.tolist()]).to(device)


@torch.no_grad()
def eval_loss(model: TinyLLM, data: torch.Tensor, args, device: torch.device, batches: int = 20) -> float:
    model.eval()
    g = torch.Generator().manual_seed(0)  # same validation windows every time
    losses = []
    for _ in range(batches):
        starts = torch.randint(0, data.numel() - args.seq_len, (args.batch_size,), generator=g)
        x = torch.stack([data[s : s + args.seq_len] for s in starts.tolist()]).to(device)
        losses.append(model(x, labels=x).loss.item())
    model.train()
    return sum(losses) / len(losses)


def sample(model: TinyLLM, tok: ByteTokenizer, prompt: str, device: torch.device, n: int = 160) -> str:
    ids = torch.tensor([tok.encode(prompt)], device=device)
    out = model.generate(ids, max_new_tokens=n, temperature=0.8, top_k=40, seed=0)
    return tok.decode(out[0].tolist())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--preset", default="tiny")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default=str(ROOT / "runs" / "tiny-shakespeare"))
    args = p.parse_args()

    torch.set_num_threads(2)
    set_seed(0)
    device = pick_device(args.device)
    tok = ByteTokenizer()
    text = load_corpus(ROOT / "data" / "tinyshakespeare.txt")
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    split = int(0.9 * data.numel())
    train, val = data[:split], data[split:]

    config = get_config(args.preset, max_seq_len=args.seq_len)
    model = TinyLLM(config).to(device)
    print(f"device={device}  params={model.num_parameters():,}  train bytes={train.numel():,}  val bytes={val.numel():,}")
    print(f"step     0  val {eval_loss(model, val, args, device):.3f}  (uniform guess = ln 256 = {math.log(256):.3f})")
    print(f"sample before training: {sample(model, tok, 'ROMEO:', device, 60)!r}")

    # Decay weights only, never norm gains (decaying a gain pulls it toward 0, not toward "no-op").
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95),
    )

    def lr_at(step: int) -> float:
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        progress = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))

    t0 = time.perf_counter()
    for step in range(1, args.steps + 1):
        for group in opt.param_groups:
            group["lr"] = lr_at(step - 1)
        x = random_batch(train, args.batch_size, args.seq_len, device)
        loss = model(x, labels=x).loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % args.eval_every == 0 or step == args.steps:
            elapsed = time.perf_counter() - t0
            tok_s = step * args.batch_size * args.seq_len / elapsed
            print(f"step {step:5d}  train {loss.item():.3f}  val {eval_loss(model, val, args, device):.3f}"
                  f"  {elapsed:6.1f}s  {tok_s:,.0f} tok/s")

    print(f"\nsample after training:\n{sample(model, tok, 'ROMEO:', device)}")
    path = save_checkpoint(model, args.out)
    print(f"\nsaved checkpoint to runs/{path.name}/")


if __name__ == "__main__":
    main()
