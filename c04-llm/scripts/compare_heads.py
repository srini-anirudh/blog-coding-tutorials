"""Train the same small model with MHA, GQA and MQA: what do you lose as you drop KV heads?

Same data, same steps, same seed. Only num_kv_heads changes. It prints parameters, cache size
per token, and validation loss. This is one seed on one small corpus, so treat small gaps as noise.

    python c04-llm/scripts/compare_heads.py --steps 400
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tinyllm import ByteTokenizer, TinyLLM, count_parameters, get_config, kv_cache_bytes  # noqa: E402
from tinyllm.utils import pick_device  # noqa: E402


@torch.no_grad()
def val_loss(model: TinyLLM, val: torch.Tensor, batch: int, seq: int, device: torch.device) -> float:
    model.eval()
    g = torch.Generator().manual_seed(0)
    losses = []
    for _ in range(20):
        starts = torch.randint(0, val.numel() - seq, (batch,), generator=g)
        x = torch.stack([val[s : s + seq] for s in starts.tolist()]).to(device)
        losses.append(model(x, labels=x).loss.item())
    model.train()
    return sum(losses) / len(losses)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = pick_device(args.device)
    path = ROOT / "data" / "tinyshakespeare.txt"
    if not path.exists():
        sys.exit("run scripts/train_sanity.py once first: it downloads the corpus")
    text = path.read_text()
    data = torch.tensor(ByteTokenizer().encode(text))
    split = int(0.9 * data.numel())
    train, val = data[:split], data[split:]

    print(f"device={device} steps={args.steps}  tiny preset, 8 query heads, seed 0")
    print(f"{'config':<14}{'kv heads':>9}{'attn params':>13}{'total params':>14}{'cache/token':>13}{'val loss':>10}{'time':>8}")
    for label, kv in [("MHA", 8), ("GQA", 4), ("GQA", 2), ("MQA", 1)]:
        torch.manual_seed(0)
        config = get_config("tiny", num_kv_heads=kv, max_seq_len=args.seq_len)
        model = TinyLLM(config).to(device).train()
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
        g = torch.Generator().manual_seed(0)  # the same batches for every config
        t0 = time.perf_counter()
        for step in range(args.steps):
            for group in opt.param_groups:
                group["lr"] = args.lr * min(1.0, (step + 1) / 100)
            starts = torch.randint(0, train.numel() - args.seq_len, (args.batch_size,), generator=g)
            x = torch.stack([train[s : s + args.seq_len] for s in starts.tolist()]).to(device)
            opt.zero_grad(set_to_none=True)
            model(x, labels=x).loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        counts = count_parameters(config)
        loss = val_loss(model, val, args.batch_size, args.seq_len, device)
        print(f"{label:<14}{kv:>9}{counts['attention']:>13,}{counts['total']:>14,}"
              f"{kv_cache_bytes(config, 1, bytes_per_value=4):>11,} B{loss:>10.3f}{time.perf_counter() - t0:>7.0f}s", flush=True)


if __name__ == "__main__":
    main()
