"""Sections 07 and 08: where the norm sits decides how the residual stream and the gradient behave with depth.

For each placement, a 24-block character LM on Tiny Shakespeare is measured at init and after a short training run:
  stream RMS     the RMS of the residual stream after blocks 0, 6, 12, 18, 24 (block 0 = the embeddings)
  rel. update    ||x_l - x_(l-1)|| / ||x_(l-1)||, how much block l changes the stream (near 0 = near identity),
                 averaged over blocks 2 to L/4 ("early") and over the last L/4 blocks ("late"). Block 1 is left out
                 because it acts on the raw embeddings, whose scale is set by the 0.02 init.
  grad ratio     ||grad of block 1 params|| / ||grad of block 24 params||
  drop cost      validation loss increase when one block is skipped, averaged over the first and last 8 blocks
The paper-scale claims (LLaMA-130M variance ~175, 7B runs) were not reproduced. This is the mechanism at toy scale.
"""

import argparse
import math
import time

import _setup  # noqa: F401
import torch

from normgeo.blocks import PLACEMENTS, TinyLM, lm_loss
from normgeo.report import Report, shakespeare

p = argparse.ArgumentParser()
p.add_argument("--layers", type=int, default=24)
p.add_argument("--d", type=int, default=64)
p.add_argument("--steps", type=int, default=400)
p.add_argument("--lr", type=float, default=2e-3)
p.add_argument("--warmup", type=int, default=50)
p.add_argument("--tag", default="")
p.add_argument("--placements", nargs="+", default=list(PLACEMENTS))
args = p.parse_args()

SEQ, BATCH = 64, 32
text = shakespeare()
chars = sorted(set(text))
data = torch.tensor([chars.index(c) for c in text])
split = int(0.9 * len(data))
train_data, val_data = data[:split], data[split:]


def batch(src: torch.Tensor, gen: torch.Generator, n: int = BATCH) -> tuple[torch.Tensor, torch.Tensor]:
    i = torch.randint(0, len(src) - SEQ - 1, (n,), generator=gen)
    x = torch.stack([src[j : j + SEQ] for j in i])
    y = torch.stack([src[j + 1 : j + SEQ + 1] for j in i])
    return x, y


VAL = batch(val_data, torch.Generator().manual_seed(123), 256)


@torch.no_grad()
def val_loss(model: TinyLM) -> float:
    model.eval()
    return lm_loss(model, *VAL).item()


def probe(model: TinyLM, xy) -> dict:
    model.train()
    model.zero_grad()
    logits, stream = model(xy[0])
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), xy[1].flatten())
    loss.backward()
    rms = [s.pow(2).mean().sqrt().item() for s in stream]
    rel = [((stream[l] - stream[l - 1]).norm() / stream[l - 1].norm()).item() for l in range(1, len(stream))]
    gnorm = [math.sqrt(sum(p.grad.pow(2).sum().item() for p in b.parameters())) for b in model.blocks]
    model.zero_grad()
    return {"rms": rms, "rel_update": rel, "grad_norm": gnorm}


def drop_costs(model: TinyLM) -> list[float]:
    base = val_loss(model)
    costs = []
    for l in range(len(model.blocks)):
        model.skip = {l}
        costs.append(val_loss(model) - base)
    model.skip = set()
    return costs


name = "06_depth" + (f"_{args.tag}" if args.tag else "")
with Report(name) as say:
    say(f"{args.layers} blocks, d = {args.d}, 4 heads, RMSNorm (DyT for 'dyt'), char-level Tiny Shakespeare, seq {SEQ}, "
        f"batch {BATCH}. AdamW lr {args.lr}, {args.warmup} warmup steps, {args.steps} steps, no weight decay.")
    marks = [0, args.layers // 4, args.layers // 2, 3 * args.layers // 4, args.layers]
    out = {}
    for stage in ["init", "trained"]:
        say()
        say(f"== {stage} ==")
        hdr = "  ".join(f"{m:>5}" for m in marks)
        say(f"{'placement':<10}stream RMS after block {hdr}   rel. update early/late   grad ratio    val loss   drop cost first/last 8")
        for pl in args.placements:
            if stage == "init":
                torch.manual_seed(0)
                model = TinyLM(len(chars), args.d, args.layers, 4, pl, SEQ)
                out[pl] = {"model": model, "init_loss": val_loss(model)}
                model_stats = probe(model, VAL)
                out[pl]["init"] = model_stats
                vl, drop = out[pl]["init_loss"], None
            else:
                model = out[pl]["model"]
                gen = torch.Generator().manual_seed(1)
                opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
                sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, args.warmup)))
                t0, curve = time.time(), []
                for step in range(args.steps):
                    model.train()
                    loss = lm_loss(model, *batch(train_data, gen))
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    sched.step()
                    curve.append(loss.item())
                out[pl]["train_seconds"] = time.time() - t0
                out[pl]["curve"] = curve[::10]
                vl = val_loss(model)
                if not math.isfinite(vl):
                    say(f"{pl:<10}diverged (val loss {vl})")
                    out[pl]["trained"] = None
                    continue
                model_stats = probe(model, VAL)
                drop = drop_costs(model)
                out[pl].update(trained=model_stats, val_loss=vl, drop_cost=drop)
            rms = "  ".join(f"{model_stats['rms'][m]:>5.2f}" for m in marks)
            q = max(2, args.layers // 4)
            r = model_stats["rel_update"]
            r = (sum(r[1:q]) / (q - 1), sum(r[-q:]) / q)
            g = model_stats["grad_norm"]
            dtxt = f"{sum(drop[:8]) / 8:>8.3f} / {sum(drop[-8:]) / 8:<8.3f}" if drop else "-"
            say(f"{pl:<10}{'':<23}{rms}   {r[0]:>8.3f} / {r[1]:<8.3f}   {g[0] / max(g[-1], 1e-12):>10.2g}   {vl:>8.3f}   {dtxt}")
    for pl in out:
        out[pl].pop("model")
    say()
    say("Notes: 'pre' and 'dyt' have no norm on the stream, so its RMS grows with depth. 'lns' divides each branch input")
    say("by sqrt(l). 'post' and 'deepnorm' renormalize the stream every sublayer, so it stays at RMS 1.")
    say.json({"args": vars(args), "results": out})
