"""Section 05: gated linear units, the 8d/3 rule, and a small-scale version of Shazeer's FFN sweep.

(a) The 8d/3 arithmetic for the configs the post quotes.
(b) Multiplicative interaction. Fit y = x1 x2 + x3 x4 + x5 x6 + x7 x8 with one FFN layer at equal parameter
    budgets. A gated unit is a degree-2 polynomial, so 4 bilinear units can represent this target exactly.
(c) A byte-level language model on wikitext-2 with each FFN variant swapped into C4's TinyLLM (attention, RoPE and
    RMSNorm come from C4 unchanged). Every variant has the same FFN parameter budget. Several seeds, so the spread
    between seeds is visible next to the gap between variants. Shazeer trained T5-base on C4 for 65k steps; this is
    a 4-layer, d=128 model trained for minutes, so it shows the mechanism, not his numbers.
(d) Statistics of the tensor that goes into W_down in each trained model: exact zeros, tails, INT8 error.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch
from torch import nn

LAB = Path(__file__).resolve().parents[1]
C4 = LAB.parents[1] / "c04-llm"
sys.path.insert(0, str(LAB))

from nonlin.data import random_windows, synthetic_bytes, wikitext_bytes  # noqa: E402
from nonlin.ffn import FFN, FFN_KINDS, ffn_params, matched_hidden, round_up  # noqa: E402
from nonlin.stats import Results, tensor_report  # noqa: E402

KINDS = ["relu", "gelu", "relu2", "bilinear", "reglu", "geglu", "swiglu"]


def part_a(res: Results) -> None:
    res.print("(a) The 8d/3 rule: gated width = (2/3) * 4d, rounded up to a multiple of 256")
    res.print(f"{'model':<12} {'d':>5} {'8d/3':>9} {'rounded':>8} {'config h':>9} {'h/d':>6}")
    rows = [("Llama-2-7B", 4096, 11008), ("Llama-3-8B", 4096, 14336), ("Mistral-7B", 4096, 14336),
            ("Qwen2.5-7B", 3584, 18944), ("Gemma-2-9B", 3584, 14336)]
    out = {}
    for name, d, h in rows:
        rule = 8 * d / 3
        res.print(f"{name:<12} {d:>5} {rule:>9.1f} {round_up(rule, 256):>8} {h:>9} {h / d:>6.2f}")
        out[name] = {"d": d, "8d/3": rule, "rounded_256": round_up(rule, 256), "config_h": h}
    d = 4096
    res.print(f"Parameters at d=4096: ungated h=4d: {ffn_params(d, 4 * d, False):,}; "
              f"gated h=11008: {ffn_params(d, 11008, True):,}")
    res.data["rule_8d3"] = out


def product_target(x: torch.Tensor) -> torch.Tensor:
    return x[:, 0] * x[:, 1] + x[:, 2] * x[:, 3] + x[:, 4] * x[:, 5] + x[:, 6] * x[:, 7]


def fit_products(kind: str, h: int, steps: int, seed: int) -> float:
    gen = torch.Generator().manual_seed(1234)
    x = torch.rand(4096, 8, generator=gen) * 2 - 1
    xt = torch.rand(4096, 8, generator=gen) * 2 - 1
    torch.manual_seed(seed)
    model = FFN(8, h, kind)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        opt.zero_grad()
        nn.functional.mse_loss(model(x)[:, 0], product_target(x)).backward()
        opt.step()
        sched.step()
    with torch.no_grad():
        return nn.functional.mse_loss(model(xt)[:, 0], product_target(xt)).item()


def part_b(res: Results, steps: int, seeds: list[int]) -> None:
    res.print(f"\n(b) y = x1x2 + x3x4 + x5x6 + x7x8, x ~ U[-1,1]^8, Var(y) = {4 / 9:.3f}. One FFN layer, Adam, "
              f"{steps} steps, held-out MSE, best of seeds {seeds}. Gated width hg, ungated width 1.5*hg "
              f"(same parameters).")
    budgets = [2, 4, 8, 16]
    res.print(f"{'kind':<9} " + " ".join(f"{'hg=' + str(h):>9}" for h in budgets))
    out = {}
    for kind in ["relu", "gelu", "relu2", "bilinear", "reglu", "swiglu"]:
        gated = FFN_KINDS[kind][1]
        row = {}
        for hg in budgets:
            h = hg if gated else int(hg * 1.5)
            row[str(hg)] = min(fit_products(kind, h, steps, s) for s in seeds)
        res.print(f"{kind:<9} " + " ".join(f"{row[str(h)]:>9.4f}" for h in budgets))
        out[kind] = row
    res.data["products"] = out


# ---------------------------------------------------------------- (c) the language model sweep
def build_lm(kind: str, d: int, layers: int, seq: int):
    sys.path.insert(0, str(C4))
    from tinyllm import ModelConfig, TinyLLM  # C4's Llama-style decoder

    h = matched_hidden(d, kind, multiple_of=8)
    cfg = ModelConfig(vocab_size=256, hidden_size=d, num_layers=layers, num_heads=4, num_kv_heads=4,
                      intermediate_size=h, max_seq_len=seq)
    model = TinyLLM(cfg)
    std = cfg.initializer_range
    for block in model.layers:
        block.mlp = FFN(d, h, kind, record=True)
        for name, lin in [("up_proj", block.mlp.up_proj), ("gate_proj", block.mlp.gate_proj),
                          ("down_proj", block.mlp.down_proj)]:
            if lin is not None:  # same init rule as C4: residual projections scaled by 1/sqrt(2L)
                nn.init.normal_(lin.weight, 0.0, std / math.sqrt(2 * layers) if name == "down_proj" else std)
    return model, h


def train_lm(job: dict) -> dict:
    torch.set_num_threads(job["threads"])
    kind, seed = job["kind"], job["seed"]
    if job["synthetic"]:
        train, val = synthetic_bytes(200_000, 0), synthetic_bytes(20_000, 1)
    else:
        train, val = wikitext_bytes("train"), wikitext_bytes("validation")
    torch.manual_seed(seed)
    model, h = build_lm(kind, job["d"], job["layers"], job["seq"])
    ffn_p = sum(p.numel() for b in model.layers for p in b.mlp.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=job["lr"], betas=(0.9, 0.95), weight_decay=0.1)
    steps, warm = job["steps"], max(1, job["steps"] // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    gen = torch.Generator().manual_seed(seed)
    t0 = time.time()
    for _ in range(steps):
        ids = random_windows(train, job["batch"], job["seq"], gen)
        loss = model(ids, labels=ids).loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
    train_s = time.time() - t0
    model.eval()
    vgen = torch.Generator().manual_seed(999)  # the same validation windows for every run
    losses, reports = [], []
    with torch.no_grad():
        for i in range(job["val_batches"]):
            ids = random_windows(val, job["batch"], job["seq"], vgen)
            losses.append(model(ids, labels=ids).loss.item())
            if i == 0:
                reports = [tensor_report(b.mlp.last_hidden) for b in model.layers]
    stats = {k: sum(r[k] for r in reports) / len(reports) for k in reports[0]}
    stats["kurtosis_max_layer"] = max(r["kurtosis"] for r in reports)
    return {"kind": kind, "seed": seed, "h": h, "ffn_params": ffn_p, "val_loss": sum(losses) / len(losses),
            "train_seconds": train_s, "down_input_stats": stats}


def part_cd(res: Results, args) -> None:
    import multiprocessing as mp

    jobs = [dict(kind=k, seed=s, d=args.d, layers=args.layers, seq=args.seq, batch=args.batch, lr=args.lr,
                 steps=args.steps, val_batches=args.val_batches, threads=args.threads, synthetic=args.synthetic)
            for k in KINDS for s in range(args.seeds)]
    with mp.get_context("spawn").Pool(args.workers) as pool:
        runs = pool.map(train_lm, jobs)
    data = "synthetic bytes" if args.synthetic else "wikitext-2 bytes"
    res.print(f"\n(c) Byte-level LM on {data}: C4 TinyLLM, d={args.d}, {args.layers} layers, 4 heads, seq {args.seq}, "
              f"batch {args.batch}, AdamW lr {args.lr}, {args.steps} steps, {args.seeds} seeds, "
              f"{args.threads} threads per run")
    res.print(f"Validation loss is nats per byte over {args.val_batches} fixed batches.")
    res.print(f"{'kind':<9} {'h':>4} {'FFN params':>11} {'val loss mean':>14} {'std':>7} {'min':>7} {'max':>7} "
              f"{'s/run':>6}")
    out = {}
    for k in KINDS:
        rs = [r for r in runs if r["kind"] == k]
        v = torch.tensor([r["val_loss"] for r in rs], dtype=torch.float64)
        std = v.std().item() if len(v) > 1 else 0.0
        secs = sum(r["train_seconds"] for r in rs) / len(rs)
        res.print(f"{k:<9} {rs[0]['h']:>4} {rs[0]['ffn_params']:>11,} {v.mean().item():>14.4f} {std:>7.4f} "
                  f"{v.min().item():>7.4f} {v.max().item():>7.4f} {secs:>6.0f}")
        stats = {key: sum(r["down_input_stats"][key] for r in rs) / len(rs) for key in rs[0]["down_input_stats"]}
        out[k] = {"h": rs[0]["h"], "ffn_params": rs[0]["ffn_params"], "val_loss": v.tolist(),
                  "val_loss_mean": v.mean().item(), "val_loss_std": std, "down_input_stats": stats}
    res.print(f"\n(d) The tensor fed into W_down, first validation batch, mean over layers and seeds")
    res.print(f"{'kind':<9} {'exact 0':>8} {'|a|<0.01rms':>12} {'kurtosis':>9} {'max kurt layer':>15} "
              f"{'max/rms':>8} {'INT8 err':>9}")
    for k in KINDS:
        s = out[k]["down_input_stats"]
        res.print(f"{k:<9} {s['exact_zero']:>8.3f} {s['near_zero_1e-2_rms']:>12.3f} {s['kurtosis']:>9.1f} "
                  f"{s['kurtosis_max_layer']:>15.1f} {s['max_over_rms']:>8.1f} {s['int8_rel_err']:>9.4f}")
    res.data["lm"] = {"settings": {k: v for k, v in jobs[0].items() if k not in ("kind", "seed")}, "by_kind": out}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--part", choices=["all", "toy", "lm"], default="all")
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--seq", type=int, default=128)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--val-batches", type=int, default=40)
    p.add_argument("--workers", type=int, default=7)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--toy-steps", type=int, default=3000)
    p.add_argument("--synthetic", action="store_true", help="offline synthetic text instead of wikitext-2")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--out", type=Path, default=LAB / "results")
    args = p.parse_args()
    if args.quick:
        args.steps, args.seeds, args.val_batches, args.toy_steps, args.workers = 20, 1, 2, 50, 2
    name = "05_ffn_variants" if args.part == "all" else f"05_ffn_variants_{args.part}"
    res = Results(name, args.out)
    if args.part in ("all", "toy"):
        part_a(res)
        part_b(res, args.toy_steps, list(range(3 if not args.quick else 1)))
    if args.part in ("all", "lm"):
        part_cd(res, args)
    res.save()


if __name__ == "__main__":
    main()
