"""A sparse mask on a dense tensor is still dense.

Run:  python c03-attention/benchmarks/bench_window.py
Part 1: the same sliding window (W = 256) through a mask and through the chunked kernel.
Part 2: block-sparse masks of falling density through the dense path. Time does not fall.
"""

from __future__ import annotations

import _common
import torch

from attnzoo.backends import attend
from attnzoo.masks import block_sparse_layout, expand_block_layout

H, D, W = 4, 32, 256


def qkv(n: int, device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return tuple(torch.randn(1, H, n, D, device=device) for _ in range(3))


@torch.inference_mode()
def main() -> None:
    torch.set_num_threads(2)
    for device_name in ["cpu", "mps"] if torch.backends.mps.is_available() else ["cpu"]:
        device = torch.device(device_name)
        print(f"\n{_common.hardware_note(device_name)}. Sliding window W = {W}, H = {H}, d = {D}. ms per call.")
        print(f"{'N':>7}{'full causal':>14}{'mask + naive':>14}{'mask + sdpa':>14}{'chunked':>10}{'scores (mask)':>16}{'scores (chunked)':>18}")
        for n in [1024, 2048, 4096, 8192] + ([16384] if device_name == "cpu" else []):
            q, k, v = qkv(n, device_name)
            full = _common.timed(lambda: attend(q, k, v, backend="sdpa", causal=True), device)
            naive = float("nan")
            if n <= 8192:
                naive = _common.timed(lambda: attend(q, k, v, backend="naive", causal=True, window_size=W), device)
            mask = _common.timed(lambda: attend(q, k, v, backend="sdpa", causal=True, window_size=W), device)
            chunked = _common.timed(lambda: attend(q, k, v, backend="window_chunked", causal=True, window_size=W), device)
            print(f"{n:>7}{full:>14.1f}{naive:>14.1f}{mask:>14.1f}{chunked:>10.1f}{n * n:>16,}{n * (2 * W - 1):>18,}", flush=True)

    print(f"\n{_common.hardware_note('cpu')}. Block-sparse masks on the dense path, N = 4096, 64 blocks. ms per call.")
    print(f"{'layout':28}{'density':>9}{'naive ms':>10}{'sdpa ms':>10}")
    device = torch.device("cpu")
    q, k, v = qkv(4096, "cpu")
    for name, local, glob in [("full causal", 64, 0), ("wide band", 16, 1), ("narrow band", 4, 1), ("near-diagonal", 1, 1)]:
        mask = expand_block_layout(block_sparse_layout(64, local, glob), 64)
        naive = _common.timed(lambda: attend(q, k, v, mask, backend="naive"), device)
        sdpa = _common.timed(lambda: attend(q, k, v, mask, backend="sdpa"), device)
        print(f"{name:28}{mask.float().mean().item():>9.3f}{naive:>10.1f}{sdpa:>10.1f}", flush=True)


if __name__ == "__main__":
    main()
