"""The memory ledger: measured state bytes of each optimizer here, then the post's arithmetic redone.

Part 1 is a measurement: build each optimizer on a small TinyLLM, take one step, count the bytes of
its state tensors. Part 2 recomputes every number the post derives for a 7B model on 8 GPUs. Those
are arithmetic from the post's stated assumptions, not measurements on a 7B run.

Run: python labs/what-an-optimizer-actually-does/scripts/06_ledger.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from optlab import SGD, Adafactor, Adam, AdamW8bit, Hybrid, Lion, Muon
from optlab.common import Report, shakespeare_batches, split_for_muon, tiny_model

GB = 1e9


def measured(rep):
    model = tiny_model()
    n = sum(p.numel() for p in model.parameters())
    hidden, rest = split_for_muon(model)
    frac = sum(p.numel() for p in hidden) / n
    rep(f"TinyLLM, hidden 128, 4 layers: {n:,} parameters; {frac:.1%} of them are hidden 2-D weights")
    makers = {
        "SGD": lambda: SGD(model.parameters(), 0.1),
        "SGD + momentum": lambda: SGD(model.parameters(), 0.1, momentum=0.9),
        "Lion": lambda: Lion(model.parameters()),
        "AdamW (fp32 m, v)": lambda: Adam(model.parameters(), weight_decay=0.1),
        "8-bit AdamW (block 2048)": lambda: AdamW8bit(model.parameters()),
        "Adafactor (b1 = 0)": lambda: Adafactor(model.parameters()),
        "Muon + AdamW hybrid": lambda: Hybrid(Muon(hidden), Adam(rest)),
        "torch.optim.AdamW": lambda: torch.optim.AdamW(model.parameters()),
    }
    batch = next(shakespeare_batches(4, 64))
    out = {}
    rep(f"{'optimizer':<26} {'state bytes / param':>20}")
    for name, make in makers.items():
        opt = make()
        model.zero_grad()
        model(batch, labels=batch).loss.backward()
        opt.step()
        if isinstance(opt, torch.optim.Optimizer):
            b = sum(v.numel() * v.element_size() for s in opt.state.values() for v in s.values()
                    if torch.is_tensor(v) and v.ndim > 0)
        else:
            b = opt.state_bytes()
        out[name] = b / n
        rep(f"{name:<26} {b / n:>20.3f}")
    rep(f"hybrid check: 4 B on {frac:.1%} + 8 B on the rest = {4 * frac + 8 * (1 - frac):.3f}")
    return out


def ledger(rep):
    P = 7e9
    rep("\n## Per-parameter totals, mixed precision: bf16 weights 2 + bf16 grads 2 + fp32 master 4 + K")
    rows = {"SGD": 0, "Adafactor": 0, "8-bit AdamW": 2, "SGD + momentum": 4, "Lion": 4, "Muon (2-D)": 4,
            "AdamW bf16 states": 4, "AdamW": 8}
    for name, k in rows.items():
        rep(f"{name:<18} K={k:>2}  total {8 + k:>2} B/param  7B model {(8 + k) * P / GB:>5.0f} GB")

    rep("\n## Sharding a 7B AdamW run over N = 8 ranks (K = 12: master 4 + m 4 + v 4)")
    K, N = 12, 8
    zero = {"replicated": 2 + 2 + K, "ZeRO-1": 2 + 2 + K / N, "ZeRO-2": 2 + (2 + K) / N, "ZeRO-3 / FSDP": (4 + K) / N}
    for name, b in zero.items():
        rep(f"{name:<14} {b:>6.3f} B/param  {b * P / GB:>6.2f} GB per rank")

    rep("\n## Checkpoint, step traffic, offload")
    ckpt = 12 * P
    rep(f"checkpoint = master + m + v = 12 B/param = {ckpt / GB:.0f} GB; bf16 weights {2 * P / GB:.0f} GB; "
        f"ratio {ckpt / (2 * P):.1f}x")
    step_bytes = 2 + 4 + 4 + 4 + 4 + 4 + 4 + 2
    rep(f"fused AdamW traffic {step_bytes} B/param, intensity 12/{step_bytes} = {12 / step_bytes:.2f} FLOP/B; "
        f"H100 ridge 989e12/3.35e12 = {989 / 3.35:.0f} FLOP/B; gap {989 / 3.35 / (12 / step_bytes):.0f}x")
    t = step_bytes * P / 3.35e12
    rep(f"7B step at 3.35 TB/s: {t * 1e3:.1f} ms unsharded, {t / 8 * 1e3:.1f} ms per rank at ZeRO-3 on 8")
    rep(f"offload: {ckpt / GB:.0f} GB over 64 GB/s PCIe = {ckpt / 64e9:.2f} s each way")

    rep("\n## Shape-dependent state")
    for m, n in ((4096, 4096), (4096, 11008)):
        rep(f"{m} x {n}: Adafactor {m * n:,} -> {m + n:,} values ({m * n / (m + n):.0f}x); "
            f"Shampoo (m^2 + n^2)/mn = {(m * m + n * n) / (m * n):.2f} floats/param")

    rep("\n## Muon cost: 5 Newton-Schulz steps on an m x n matrix, n = min(m, n)")
    T = 4e6
    for m, n in ((4096, 4096), (11008, 4096)):
        flops = 5 * (2 * n * n * m + 2 * n**3 + 2 * n * n * m)
        per = flops / (m * n)
        rep(f"{m} x {n}: {per:,.0f} FLOP/param (20n + 10n^2/m = {20 * n + 10 * n * n / m:,.0f}); "
            f"vs 6T at T = 4M tokens: {per / (6 * T):.2%}")
    rep(f"post's shortcut 3.3 d/T with d = 4096: {3.3 * 4096 / T:.2%} (drops the n^2/m term)")
    for frac in (0.70, 0.85, 13e9 / 2 / P):
        b = 2 * frac * P
        rep(f"Muon gather, 2 B/param on {frac:.0%} hidden: {b / GB:.1f} GB/step; "
            f"NVLink 400 GB/s {b / 400e9 * 1e3:.0f} ms, EFA 40 GB/s {b / 40e9 * 1e3:.0f} ms")
    return zero


def main():
    rep = Report("06_ledger")
    rep("## Measured optimizer state on a small model (one step taken)")
    m = measured(rep)
    z = ledger(rep)
    rep.save({"measured_state_bytes_per_param": m, "zero_bytes_per_param": z})


if __name__ == "__main__":
    main()
