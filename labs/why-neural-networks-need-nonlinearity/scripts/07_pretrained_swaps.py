"""Section 08 (and 05, 06) on real checkpoints: what happens when the activation does not match the weights.

(a) GPT-2 small was trained with the tanh form of GELU. Swap in exact GELU, QuickGELU, SiLU and ReLU without any
    fine-tuning and measure wikitext-2 test perplexity.
(b) OpenAI CLIP ViT-B/32 was trained with QuickGELU. Swap plain GELU into the vision tower only and measure
    zero-shot CIFAR-10 accuracy. The post's example is ViT-L/14 on ImageNet; this is the smaller model on a
    smaller test, so it shows the size of the effect at this scale only.
(c) The tensor that goes into W_down in two real models: GPT-2 (elementwise GELU) and SmolLM2-135M (SwiGLU).
    They differ in more than the FFN (data, size, norm, training), so this is an observation, not an ablation.

Needs the Hugging Face Hub, or the models already in HF_HOME: gpt2, openai/clip-vit-base-patch32,
HuggingFaceTB/SmolLM2-135M, and the datasets Salesforce/wikitext and uoft-cs/cifar10.
"""

from __future__ import annotations

import argparse
import io
import math
import sys
from pathlib import Path

import torch

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))

from nonlin import activations as A  # noqa: E402
from nonlin.data import wikitext_bytes  # noqa: E402
from nonlin.stats import Results, tensor_report  # noqa: E402

CIFAR10 = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


class Fn(torch.nn.Module):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.fn = A.get(name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(x)


def wikitext_text(split: str) -> str:
    return wikitext_bytes(split).numpy().tobytes().decode("utf-8")


@torch.no_grad()
def gpt2_perplexity(model, ids: torch.Tensor, window: int) -> float:
    nll, count = 0.0, 0
    for s in range(0, len(ids) - window + 1, window):
        chunk = ids[s : s + window].unsqueeze(0)
        loss = model(chunk, labels=chunk).loss.item()
        nll += loss * (window - 1)
        count += window - 1
    return math.exp(nll / count)


def part_a(res: Results, windows: int, window: int) -> None:
    from transformers import AutoTokenizer, GPT2LMHeadModel

    tok = AutoTokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").eval()
    ids = tok(wikitext_text("test"), return_tensors="pt").input_ids[0][: windows * window]
    res.print(f"(a) GPT-2 small (config activation: {model.config.activation_function}), wikitext-2 test, "
              f"{windows} non-overlapping windows of {window} tokens, no fine-tuning")
    original = [blk.mlp.act for blk in model.transformer.h]
    probe = ids[:64].unsqueeze(0)
    ref_logits = model(probe).logits
    out = {}
    res.print(f"{'activation in every MLP':<24} {'perplexity':>11} {'max |logit change|, 64 tokens':>31}")
    for name in ["gelu_tanh", "gelu", "quick_gelu", "silu", "relu"]:
        for blk in model.transformer.h:
            blk.mlp.act = Fn(name)
        with torch.no_grad():
            dlog = (model(probe).logits - ref_logits).abs().max().item()
        ppl = gpt2_perplexity(model, ids, window)
        res.print(f"{name:<24} {ppl:>11.2f} {dlog:>31.3g}")
        out[name] = {"perplexity": ppl, "max_logit_change": dlog}
    for blk, act in zip(model.transformer.h, original):
        blk.mlp.act = act
    res.data["gpt2"] = out


def cifar_test(n: int):
    import os

    from PIL import Image

    home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    files = sorted((home / "hub" / "datasets--uoft-cs--cifar10" / "snapshots").glob("*/plain_text/test-*.parquet"))
    if files:
        import pyarrow.parquet as pq

        tbl = pq.read_table(files[0]).slice(0, n).to_pydict()
        imgs = [Image.open(io.BytesIO(d["bytes"])).convert("RGB") for d in tbl["img"]]
        return imgs, tbl["label"]
    from datasets import load_dataset

    ds = load_dataset("uoft-cs/cifar10", split=f"test[:{n}]")
    return [im.convert("RGB") for im in ds["img"]], ds["label"]


@torch.no_grad()
def part_b(res: Results, n: int, batch: int) -> None:
    from transformers import CLIPModel, CLIPProcessor

    name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(name).eval()
    proc = CLIPProcessor.from_pretrained(name)
    imgs, labels = cifar_test(n)
    labels = torch.tensor(labels)
    text = proc(text=[f"a photo of a {c}." for c in CIFAR10], return_tensors="pt", padding=True)
    t_emb = model.get_text_features(**text)
    t_emb = t_emb.pooler_output if hasattr(t_emb, "pooler_output") else t_emb
    t_emb = t_emb / t_emb.norm(dim=-1, keepdim=True)
    pixels = [proc(images=imgs[i : i + batch], return_tensors="pt").pixel_values for i in range(0, n, batch)]
    layers = model.vision_model.encoder.layers
    res.print(f"\n(b) CLIP ViT-B/32 (vision hidden_act: {model.config.vision_config.hidden_act}), zero-shot CIFAR-10, "
              f"first {n} test images, prompt 'a photo of a {{label}}.'; only the vision tower's activation changes")
    res.print(f"{'vision activation':<18} {'accuracy':>9} {'mean cos to original embedding':>31}")
    out, ref = {}, None
    for act in ["quick_gelu", "gelu", "gelu_tanh"]:
        for layer in layers:
            layer.mlp.activation_fn = Fn(act)
        embs = []
        for px in pixels:
            e = model.get_image_features(pixel_values=px)
            e = e.pooler_output if hasattr(e, "pooler_output") else e
            embs.append(e / e.norm(dim=-1, keepdim=True))
        emb = torch.cat(embs)
        if ref is None:
            ref = emb
        acc = ((emb @ t_emb.T).argmax(1) == labels).float().mean().item()
        cos = (emb * ref).sum(1).mean().item()
        res.print(f"{act:<18} {acc:>9.4f} {cos:>31.4f}")
        out[act] = {"accuracy": acc, "mean_cos_to_quick_gelu": cos, "n_images": n}
    res.data["clip"] = out


@torch.no_grad()
def part_c(res: Results, n_tokens: int) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    text = wikitext_text("validation")
    res.print(f"\n(c) The input to W_down in real checkpoints, first {n_tokens} tokens of wikitext-2 validation, "
              f"mean over layers")
    res.print(f"{'model (FFN)':<36} {'exact 0':>8} {'|a|<0.01rms':>12} {'kurtosis':>9} {'max kurt layer':>15} "
              f"{'max/rms':>8} {'INT8 err':>9}")
    out = {}
    for name, label in [("gpt2", "GPT-2 (GELU)"), ("HuggingFaceTB/SmolLM2-135M", "SmolLM2-135M (SwiGLU)")]:
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()
        ids = tok(text, return_tensors="pt").input_ids[:, :n_tokens]
        grabbed: list[torch.Tensor] = []
        if name == "gpt2":
            targets = [blk.mlp.c_proj for blk in model.transformer.h]
        else:
            targets = [layer.mlp.down_proj for layer in model.model.layers]
        hooks = [m.register_forward_pre_hook(lambda _m, inp: grabbed.append(inp[0].detach())) for m in targets]
        model(ids)
        for hk in hooks:
            hk.remove()
        # Position 0 carries the "massive activations" of Sun et al. (2024), so report with and without it.
        for suffix, start in [("", 0), (", no token 0", 1)]:
            reps = [tensor_report(g[:, start:]) for g in grabbed]
            s = {k: sum(r[k] for r in reps) / len(reps) for k in reps[0]}
            s["kurtosis_max_layer"] = max(r["kurtosis"] for r in reps)
            row = label + suffix
            res.print(f"{row:<36} {s['exact_zero']:>8.3f} {s['near_zero_1e-2_rms']:>12.3f} {s['kurtosis']:>9.1f} "
                      f"{s['kurtosis_max_layer']:>15.1f} {s['max_over_rms']:>8.1f} {s['int8_rel_err']:>9.4f}")
            out[row] = s
    res.data["real_down_inputs"] = out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--windows", type=int, default=40)
    p.add_argument("--window", type=int, default=1024)
    p.add_argument("--images", type=int, default=2000)
    p.add_argument("--tokens", type=int, default=1024)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--out", type=Path, default=LAB / "results")
    args = p.parse_args()
    if args.quick:
        args.windows, args.window, args.images, args.tokens = 2, 128, 32, 128
    res = Results("07_pretrained_swaps", args.out)
    part_a(res, args.windows, args.window)
    part_b(res, args.images, 100)
    part_c(res, args.tokens)
    res.save()


if __name__ == "__main__":
    main()
