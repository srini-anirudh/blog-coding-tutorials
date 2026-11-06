"""The simplest pretraining loop that works. Everything else in this tutorial fixes what it lacks.

    python c05-pretraining/scripts/naive_train.py --steps 200
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from pretrain.config import ModelArgs  # noqa: E402
from pretrain.model import build_model  # noqa: E402
from pretrain.utils import resolve  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    torch.manual_seed(0)
    L, B = 64, 8
    tokens = np.memmap(resolve("data_cache/tinystories/train_000.bin"), dtype=np.uint16, mode="r")
    model = build_model(ModelArgs(hidden_size=64, num_layers=2, num_heads=4, num_kv_heads=2,
                                  intermediate_size=172, max_seq_len=L))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        # A random start per row. No epochs, no position to save, no sharding.
        starts = torch.randint(0, len(tokens) - L - 1, (B,)).tolist()
        chunk = torch.stack([torch.from_numpy(tokens[s : s + L + 1].astype(np.int64)) for s in starts])
        x, y = chunk[:, :-1], chunk[:, 1:]  # (B, L) each
        logits = model(x).logits  # (B, L, V)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        loss.backward()
        opt.step()
        opt.zero_grad()
        if step == 1 or step % (args.steps // 5) == 0:
            print(f"step {step:4d}  loss {loss.item():.3f}  {step * B * L / (time.time() - t0):,.0f} tok/s")


if __name__ == "__main__":
    main()
