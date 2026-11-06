"""Why backward slows down as KV heads fall: enable_gqa against repeat_kv, on PyTorch's CPU flash kernel.

Run:  python c03-attention/benchmarks/bench_backward.py
Writes results/bench_backward.txt and prints the same lines. CPU only, fp32, causal.
Part 1: the SDPA kernel alone, across KV head counts and thread counts, B = 1.
Part 2: one attnzoo layer across batch sizes, 8 threads.
Part 3: one attnzoo layer at N = 8192, 8 threads, B = 1.
The paths in a table take turns, so load that drifts during a run hits them alike. The first round
is a warm-up and is not counted. Cells show the median and, in brackets, the fastest and slowest call.
"""

from __future__ import annotations

import os
import statistics
import time
from pathlib import Path
from typing import Callable

import _common
import torch
import torch.nn.functional as F

from attnzoo.attention_sdpa import sdpa_attention
from attnzoo.backends import BACKENDS
from attnzoo.config import AttentionConfig
from attnzoo.heads import repeat_kv
from attnzoo.library import build_attention

D, HQ, DH = 256, 8, 32
REPEATS = 5
INIT_STD = 0.57  # about the std of q, k and v from freshly initialised Linear layers
LAYERS = {"GQA Hkv=2": ("gqa", 2), "MQA": ("mqa", 1)}
OUT = Path(__file__).resolve().parent.parent / "results" / "bench_backward.txt"
Step = Callable[[], tuple[float, float]]  # one forward and backward, returns (forward ms, backward ms)


def sdpa_repeat(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: torch.Tensor | None = None, **kw) -> torch.Tensor:
    """The sdpa backend, with K and V copied up to Hq heads so that enable_gqa stays off."""
    n_rep = q.size(1) // k.size(1)
    return sdpa_attention(q, repeat_kv(k, n_rep), repeat_kv(v, n_rep), attn_mask, **kw)


BACKENDS["sdpa_repeat"] = sdpa_repeat  # local to this process; a new way to run attention is one dict entry


def kernel_step(n: int, hkv: int, use_flag: bool) -> Step:
    """The kernel with nothing around it: q, k and v are leaves, fresh on every call. B = 1."""

    def leaf(heads: int) -> torch.Tensor:
        return (torch.randn(1, heads, n, DH) * INIT_STD).requires_grad_()

    def step() -> tuple[float, float]:
        q, k, v = leaf(HQ), leaf(hkv), leaf(hkv)
        n_rep = HQ // hkv
        t0 = time.perf_counter()
        if use_flag:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        else:
            out = F.scaled_dot_product_attention(q, repeat_kv(k, n_rep), repeat_kv(v, n_rep), is_causal=True)
        loss = out.pow(2).mean()
        t1 = time.perf_counter()
        loss.backward()
        return (t1 - t0) * 1e3, (time.perf_counter() - t1) * 1e3

    return step


def make_layer(kind: str, hkv: int, backend: str, n: int) -> torch.nn.Module:
    """The same seed gives every path the same weights."""
    torch.manual_seed(0)
    return build_attention(AttentionConfig(kind, D, HQ, num_kv_heads=hkv, max_seq_len=n + 1, backend=backend))


def layer_step(kind: str, hkv: int, backend: str, b: int, n: int) -> Step:
    """One full layer on the same input every call, with a loss like bench_zoo's."""
    model = make_layer(kind, hkv, backend, n)
    x, target = torch.randn(b, n, D), torch.randn(b, n, D)

    def step() -> tuple[float, float]:
        t0 = time.perf_counter()
        loss = (model(x) - target).pow(2).mean()
        t1 = time.perf_counter()
        loss.backward()
        bwd = (time.perf_counter() - t1) * 1e3
        model.zero_grad(set_to_none=True)
        return (t1 - t0) * 1e3, bwd

    return step


def measure(paths: dict[str, Step], repeats: int = REPEATS) -> dict[str, dict[str, list[float]]]:
    """Run the paths in turn for repeats + 1 rounds and keep every round after the first."""
    times = {name: {"fwd": [], "bwd": []} for name in paths}
    for round_index in range(repeats + 1):
        for name, step in paths.items():
            fwd, bwd = step()
            if round_index > 0:
                times[name]["fwd"].append(fwd)
                times[name]["bwd"].append(bwd)
    return times


def med(values: list[float]) -> float:
    return statistics.median(values)


def cell(values: list[float]) -> str:
    return f"{med(values):.1f} [{min(values):.1f}-{max(values):.1f}]"


def main() -> None:
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w") as f:

        def say(line: str = "") -> None:
            print(line, flush=True)
            f.write(line + "\n")
            f.flush()

        say(f"# {_common.hardware_note('cpu')}. CPU flash kernel, fp32, causal.")
        say(f"# Load average at the start: {os.getloadavg()[0]:.0f}. The machine is shared, so read the brackets.")
        say(f"# Median [fastest-slowest] of {REPEATS} calls after a warm-up, in ms. flag = enable_gqa=True, repeat = repeat_kv.")
        say("# bench_zoo.py also uses B = 1.")

        n, threads = 4096, [1, 2, 4, 8]
        say(f"\nPart 1: the SDPA kernel alone, backward ms. B = 1, Hq = {HQ}, N = {n}, d = {DH}.")
        paths = [("Hkv=8 (no flag)", 8, False)]
        for hkv in (4, 2, 1):
            paths += [(f"Hkv={hkv} flag", hkv, True), (f"Hkv={hkv} repeat", hkv, False)]
        say(f"{'path':18}" + "".join(f"{'threads = ' + str(t):>24}" for t in threads) + f"{'speed-up, 1 to 8':>18}")
        rows: dict[str, list[list[float]]] = {name: [] for name, _, _ in paths}
        for t in threads:
            torch.set_num_threads(t)
            times = measure({name: kernel_step(n, hkv, flag) for name, hkv, flag in paths})
            for name in rows:
                rows[name].append(times[name]["bwd"])
        for name, per_thread in rows.items():
            speedup = med(per_thread[0]) / med(per_thread[-1])
            say(f"{name:18}" + "".join(f"{cell(v):>24}" for v in per_thread) + f"{speedup:>18.2f}")

        torch.set_num_threads(8)
        n = 2048
        say(f"\nPart 2: one attnzoo layer, backward ms, by batch size. 8 threads, N = {n}, D = {D}, Hq = {HQ}.")
        say(f"{'B':>3}  {'layer':11}{'flag':>28}{'repeat':>28}{'flag / repeat':>15}")
        for b in (1, 2, 4, 8):
            mha = measure({"mha": layer_step("mha", 8, "sdpa", b, n)})
            say(f"{b:>3}  {'MHA':11}{cell(mha['mha']['bwd']):>28}{'no flag':>28}{'-':>15}")
            for name, (kind, hkv) in LAYERS.items():
                times = measure({"flag": layer_step(kind, hkv, "sdpa", b, n), "repeat": layer_step(kind, hkv, "sdpa_repeat", b, n)})
                flag, repeat = times["flag"]["bwd"], times["repeat"]["bwd"]
                say(f"{b:>3}  {name:11}{cell(flag):>28}{cell(repeat):>28}{med(flag) / med(repeat):>15.2f}")

        n = 8192
        say(f"\nPart 3: one attnzoo layer. B = 1, 8 threads, N = {n}, D = {D}, Hq = {HQ}.")
        say(f"{'layer':11}{'path':9}{'forward ms':>28}{'backward ms':>28}")
        mha = measure({"mha": layer_step("mha", 8, "sdpa", 1, n)})
        say(f"{'MHA':11}{'no flag':9}{cell(mha['mha']['fwd']):>28}{cell(mha['mha']['bwd']):>28}")
        for name, (kind, hkv) in LAYERS.items():
            times = measure({"flag": layer_step(kind, hkv, "sdpa", 1, n), "repeat": layer_step(kind, hkv, "sdpa_repeat", 1, n)})
            for path in ("flag", "repeat"):
                say(f"{name:11}{path:9}{cell(times[path]['fwd']):>28}{cell(times[path]['bwd']):>28}")

        n = 1024
        say(f"\nSame answer? One layer, B = 1, N = {n}, flag against repeat, max |difference|:")
        x = torch.randn(1, n, D)
        for name, (kind, hkv) in LAYERS.items():
            a, b = make_layer(kind, hkv, "sdpa", n), make_layer(kind, hkv, "sdpa_repeat", n)
            out_a, out_b = a(x), b(x)
            out_a.pow(2).mean().backward()
            out_b.pow(2).mean().backward()
            grad = max((pa.grad - pb.grad).abs().max().item() for pa, pb in zip(a.parameters(), b.parameters()))
            say(f"  {name:11}output {(out_a - out_b).abs().max().item():.1e}, parameter gradients {grad:.1e}")


if __name__ == "__main__":
    main()
