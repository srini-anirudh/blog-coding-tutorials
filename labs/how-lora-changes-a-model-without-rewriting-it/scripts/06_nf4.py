"""Quantization from first principles (post: "QLoRA: training through a compressed model", Fig. 7).

1. Build NF4 from normal quantiles and compare it with the published codebook.
2. Gaussian weights: NF4 vs 16 evenly spaced levels, per-tensor vs blocks of 64, with and without outliers.
3. Double quantization: count the bits actually stored, and what quantizing the constants costs in accuracy.
4. Real weights: the same errors on every linear layer of SmolLM2-135M, and its WikiText-2 loss in fp32,
   with NF4 linears, and with uniform 4-bit linears (--no-model skips this part).
"""

import argparse
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from loralab.lora import ALL_LINEAR
from loralab.quant import (NF4_PUBLISHED, double_dequantize, double_quantize, dequantize_blockwise, nf4_codebook,
                           quantize_blockwise, quantize_model_linears, storage_bits_per_param, uniform_codebook)
from loralab.report import Report, table


def rel_err(w, codebook, block):
    codes, absmax = quantize_blockwise(w, codebook, block)
    wq = dequantize_blockwise(codes, absmax, codebook, w.shape)
    return ((wq - w).norm() / w.norm()).item(), codes


def code_entropy(codes) -> float:
    p = torch.bincount(codes.view(-1).long(), minlength=16).float()
    p = p[p > 0] / p.sum()
    return -(p * p.log2()).sum().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-model", action="store_true", help="skip the SmolLM2-135M part")
    ap.add_argument("--rows", type=int, default=64, help="WikiText-2 test rows of 512 tokens")
    args = ap.parse_args()
    rep = Report("06_nf4")
    nf4, uni = NF4_PUBLISHED, uniform_codebook(16)

    built = nf4_codebook()
    rep("## Part 1: NF4 built from normal quantiles")
    rep("built:", " ".join(f"{v:+.4f}" for v in built.tolist()))
    rep(f"max |built - published| = {(built - nf4).abs().max().item():.1e}; zero is a codeword: {bool((built == 0).any())}")
    rep(f"16 evenly spaced levels include zero: {bool((uni == 0).any())}")
    rep.data["codebook_max_diff"] = (built - nf4).abs().max().item()

    rep("\n## Part 2: 4096 x 4096 Gaussian weights, std 0.02, seed 0")
    g = torch.Generator().manual_seed(0)
    w = torch.randn(4096, 4096, generator=g) * 0.02
    w_out = w.clone()
    idx = torch.randperm(w.numel(), generator=g)[: w.numel() // 1000]
    w_out.view(-1)[idx] *= 20          # 0.1% of weights become 20x outliers
    rows = []
    for label, ww in (("Gaussian", w), ("Gaussian + 0.1% outliers x20", w_out)):
        for block in (None, 4096, 64):
            for name, cb in (("NF4", nf4), ("uniform 16", uni)):
                e, codes = rel_err(ww, cb, block)
                rows.append([label, "whole tensor" if block is None else block, name, e, code_entropy(codes)])
    for line in table(rows, ["weights", "block", "codebook", "relative error", "code entropy (bits, max 4)"]):
        rep(line)
    rep.data["gaussian"] = rows

    rep("\n## Part 3: double quantization of the absmax constants (blocks of 64, constants in blocks of 256)")
    n = w.numel()
    b1, b2 = storage_bits_per_param(n, double_quant=False), storage_bits_per_param(n, double_quant=True)
    rep(f"stored bits per parameter, counted from tensor sizes: {b1:.4f} -> {b2:.4f} (saving {b1 - b2:.4f})")
    codes, absmax = quantize_blockwise(w, nf4, 64)
    absmax_dq = double_dequantize(*double_quantize(absmax))
    e1 = ((dequantize_blockwise(codes, absmax, nf4, w.shape) - w).norm() / w.norm()).item()
    e2 = ((dequantize_blockwise(codes, absmax_dq, nf4, w.shape) - w).norm() / w.norm()).item()
    rep(f"relative error of the constants after int8: {((absmax_dq - absmax).abs() / absmax).max().item():.2e} (max)")
    rep(f"weight error: fp32 constants {e1:.5f}, int8 constants {e2:.5f}")
    rep.data["double_quant"] = {"bits_single": b1, "bits_double": b2, "err_single": e1, "err_double": e2}

    if args.no_model:
        rep.save()
        return
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from loralab.data import MODEL_ID, chunks, mean_loss, wikitext
    rep(f"\n## Part 4: {MODEL_ID}, every q,k,v,o,gate,up,down weight (blocks of 64)")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    num, den = {"NF4": 0.0, "uniform 16": 0.0}, 0.0
    kurt = []
    for name, mod in model.named_modules():
        if name.split(".")[-1] in ALL_LINEAR:
            ww = mod.weight.detach()
            den += ww.pow(2).sum().item()
            for cname, cb in (("NF4", nf4), ("uniform 16", uni)):
                codes, absmax = quantize_blockwise(ww, cb, 64)
                num[cname] += (dequantize_blockwise(codes, absmax, cb, ww.shape) - ww).pow(2).sum().item()
            z = (ww - ww.mean()) / ww.std()
            kurt.append(z.pow(4).mean().item())
    errs = {k: math.sqrt(v / den) for k, v in num.items()}
    rep(f"relative error over all linear weights: NF4 {errs['NF4']:.4f}, uniform 16 {errs['uniform 16']:.4f}")
    rep(f"kurtosis of the weights (Gaussian = 3): median {sorted(kurt)[len(kurt) // 2]:.2f}, max {max(kurt):.1f}")
    rows_test = chunks(tok, wikitext("test"), 512, limit=args.rows)
    losses = {"fp32": mean_loss(model, rows_test)}
    for cname, cb in (("NF4", nf4), ("uniform 16", uni)):
        qm = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
        if cname == "NF4":
            quantize_model_linears(qm, ALL_LINEAR)
        else:
            for name, mod in qm.named_modules():
                if name.split(".")[-1] in ALL_LINEAR:
                    c, a = quantize_blockwise(mod.weight, cb, 64)
                    mod.weight.data = dequantize_blockwise(c, a, cb, mod.weight.shape)
        losses[cname] = mean_loss(qm, rows_test)
        del qm
    rep(f"WikiText-2 test, {len(rows_test)} rows x 512 tokens, mean loss (perplexity):")
    for k, v in losses.items():
        rep(f"  {k:>10}: {v:.4f} ({math.exp(v):.2f})")
    rep.data["smollm2"] = {"weight_rel_err": errs, "kurtosis_median": sorted(kurt)[len(kurt) // 2],
                           "kurtosis_max": max(kurt), "wikitext_loss": losses, "rows": len(rows_test)}
    rep.save()


if __name__ == "__main__":
    main()
