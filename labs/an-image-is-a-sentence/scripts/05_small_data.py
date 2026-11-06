"""Post sections: The missing inductive bias, DeiT and the data problem, ConViT.

Train four small models from scratch on CIFAR-10 subsets of 1,000, 5,000 and 50,000 images, with the same number
of optimiser steps at every size:
  cnn     TinyCNN: locality, weight sharing and hierarchy built in
  vit     TinyViT with a class token
  convit  TinyViT whose first 4 of 6 blocks are ConViT GPSA, gates initialised at sigmoid(1) = 0.73 positional
  deit    TinyViT with a distillation token, taught by the cnn trained on the same subset (hard labels)
The post's data threshold (10 to 30 M images) is far beyond CIFAR; this shows the small-data side only.

    python labs/an-image-is-a-sentence/scripts/05_small_data.py --device cuda          # the run in results/
    python labs/an-image-is-a-sentence/scripts/05_small_data.py --quick                # a 1-minute CPU smoke run
Saves results/05_small_data.txt and .json (results/05_small_data_quick.* with --quick).
"""

import argparse
import math
import time

import _setup  # noqa: F401
import torch
import torch.nn.functional as F

from vitfamily.c03 import be_polite
from vitfamily.data import augment, cifar10, normalize
from vitfamily.models import TinyCNN, TinyViT, n_params
from vitfamily.report import Report

p = argparse.ArgumentParser()
p.add_argument("--device", default="cpu")
p.add_argument("--sizes", type=int, nargs="+", default=[1000, 5000, 50000])
p.add_argument("--steps", type=int, default=3000)
p.add_argument("--batch", type=int, default=128)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--dim", type=int, default=128)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--quick", action="store_true", help="sizes 500, steps 60, 1,000 test images")
args = p.parse_args()
if args.quick:
    args.sizes, args.steps = [500], 60
be_polite(8)
dev = torch.device(args.device)
log = Report("05_small_data_quick" if args.quick else "05_small_data", device=dev.type)

xtr, ytr = cifar10("train")
xte, yte = cifar10("test")
if args.quick:
    xte, yte = xte[:1000], yte[:1000]
xtr, xte = normalize(xtr).to(dev), normalize(xte).to(dev)
ytr, yte = ytr.to(dev), yte.to(dev)
amp = torch.autocast("cuda", dtype=torch.bfloat16) if dev.type == "cuda" else torch.autocast("cpu", enabled=False)


def build(kind: str) -> torch.nn.Module:
    if kind == "cnn":
        return TinyCNN(width=48)
    vit = dict(dim=args.dim, depth=6, heads=4)
    return {"vit": lambda: TinyViT(**vit), "convit": lambda: TinyViT(**vit, gpsa_layers=4),
            "deit": lambda: TinyViT(**vit, distill=True)}[kind]()  # fmt: skip


def lr_at(step: int) -> float:
    warm = args.steps // 20
    if step < warm:
        return args.lr * (step + 1) / warm
    return args.lr * 0.5 * (1 + math.cos(math.pi * (step - warm) / (args.steps - warm)))


@torch.no_grad()
def evaluate(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    model.eval()
    outs: dict[str, list] = {}
    for i in range(0, len(xte), 1000):
        with amp:
            o = model(xte[i : i + 1000])
        for k, v in o.items():
            outs.setdefault(k, []).append(v.float())
    model.train()
    return {k: torch.cat(v) for k, v in outs.items()}


def train(kind: str, idx: torch.Tensor, teacher: torch.nn.Module | None = None) -> torch.nn.Module:
    torch.manual_seed(args.seed)
    model = build(kind).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    for step in range(args.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        b = idx[torch.randint(len(idx), (args.batch,), generator=gen)].to(dev)
        x, y = augment(xtr[b]), ytr[b]
        with amp:
            out = model(x)
            if kind == "deit":
                with torch.no_grad():
                    y_teacher = teacher(x)["logits"].argmax(-1)
                loss = 0.5 * F.cross_entropy(out["logits_cls"], y) + 0.5 * F.cross_entropy(out["logits_dist"], y_teacher)
            else:
                loss = F.cross_entropy(out["logits"], y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return model


def acc(logits: torch.Tensor) -> float:
    return (logits.argmax(-1) == yte).float().mean().item()


log(f"CIFAR-10, test set {len(yte)} images. {args.steps} AdamW steps of batch {args.batch} at every size, "
    f"lr {args.lr} cosine, wd 0.05, crop + flip. Seed {args.seed}, one run per cell.")
log(f"params: cnn {n_params(build('cnn')):,}  vit {n_params(build('vit')):,}  convit {n_params(build('convit')):,}  "
    f"deit {n_params(build('deit')):,}")
results = {}
for n in args.sizes:
    idx = torch.randperm(len(ytr), generator=torch.Generator().manual_seed(args.seed))[:n]
    row, t0 = {}, time.time()
    cnn = train("cnn", idx).eval()
    row["cnn"] = acc(evaluate(cnn)["logits"])
    row["vit"] = acc(evaluate(train("vit", idx))["logits"])
    convit = train("convit", idx)
    row["convit"] = acc(evaluate(convit)["logits"])
    row["convit_gates"] = [
        [round(v, 3) for v in torch.sigmoid(b.attn.gate).tolist()] for b in convit.blocks[: convit.gpsa_layers]
    ]
    deit = train("deit", idx, teacher=cnn)
    o = evaluate(deit)
    row["deit"] = acc(o["logits"])
    row["deit_cls_head"], row["deit_dist_head"] = acc(o["logits_cls"]), acc(o["logits_dist"])
    row["deit_head_disagreement"] = (o["logits_cls"].argmax(-1) != o["logits_dist"].argmax(-1)).float().mean().item()
    row["deit_cos_final"] = F.cosine_similarity(o["cls_feat"], o["dist_feat"]).mean().item()
    tok = deit.special[0].float()
    row["deit_cos_input_tokens"] = F.cosine_similarity(tok[0], tok[1], dim=0).item()
    row["seconds"] = round(time.time() - t0, 1)
    results[n] = row
    log(f"\n== {n} training images ({row['seconds']} s) ==")
    log(f"test accuracy: cnn {row['cnn']:.3f}  vit {row['vit']:.3f}  convit {row['convit']:.3f}  deit {row['deit']:.3f}")
    log(f"deit heads: class token {row['deit_cls_head']:.3f}, distillation token {row['deit_dist_head']:.3f}, "
        f"disagree on {100 * row['deit_head_disagreement']:.1f}% of test images")
    log(f"deit cosine(class, distillation): learned input tokens {row['deit_cos_input_tokens']:.3f}, "
        f"final embeddings {row['deit_cos_final']:.3f}")
    log(f"convit gate sigmoid(lambda) per head after training (init 0.731), blocks 1-4: {row['convit_gates']}")

log("\n== Summary: test accuracy by training-set size ==")
log(f"{'images':>8}{'cnn':>8}{'vit':>8}{'convit':>8}{'deit':>8}")
for n, r in results.items():
    log(f"{n:>8}{r['cnn']:>8.3f}{r['vit']:>8.3f}{r['convit']:>8.3f}{r['deit']:>8.3f}")
log.data = {"args": vars(args), "results": results}
log.save()
