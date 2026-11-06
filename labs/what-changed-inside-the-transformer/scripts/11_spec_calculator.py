"""Build a model: the post's Figure 10 calculator in Python, checked against exact parameter counts.

Post section: "Build a model". The formulas are ported from the post's figures.js (organs/arith.py, calc_model).
With --hub, each preset's real config.json is downloaded from the Hugging Face Hub (a few KB each, no weights)
and the model is built on the meta device to count its parameters exactly.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from organs.arith import GIB, KIB, PRESETS, calc_model
from organs.report import Report

HUB_IDS = {  # public repos with the same config as each preset (the Llama ones are ungated mirrors)
    "gpt2xl": "openai-community/gpt2-xl",
    "llama2": "NousResearch/Llama-2-7b-hf",
    "llama3": "NousResearch/Meta-Llama-3-8B",
    "olmo3": "allenai/Olmo-3-1125-32B",
    "qwen3": "Qwen/Qwen3-235B-A22B",
    "gptoss": "openai/gpt-oss-120b",
    "dsv3": "deepseek-ai/DeepSeek-V3",
}


def exact_count(repo: str) -> int:
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(repo)
    if getattr(cfg, "quantization_config", None) is not None:
        del cfg.quantization_config  # count parameters, not quantized storage
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg)
    return sum(p.numel() for p in model.parameters())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hub", action="store_true", help="compare with exact counts from Hub configs (network)")
    args = ap.parse_args()
    rep = Report("11_spec_calculator" + ("_hub" if args.hub else ""))
    rep("Figure 10 formulas. P_attn = 2 L d d_h (n_q + n_kv) (MLA: its own projections), P_ffn = 3 L d d_ff or")
    rep("3 L d d_e (N + shared), FLOPs/token = 2 P_active + 4 L n_q d_h s. KV in bf16. Norms not counted.")
    rows = []
    for key, spec in PRESETS.items():
        m = calc_model(spec)
        row = [spec.label, f"{m['total'] / 1e9:.2f}B", f"{m['active'] / 1e9:.2f}B", f"{m['kv_per_tok'] / KIB:.1f} KiB",
               f"{spec.s:,}", f"{m['kv_at_s'] / GIB:.2f} GiB", f"{m['flops'] / 1e9:.1f}", f"{m['attn_share']:.1%}"]
        if args.hub:
            n = exact_count(HUB_IDS[key])
            row += [f"{n / 1e9:.3f}B", f"{(m['total'] - n) / n:+.2%}"]
        rows.append(row)
        print(f"done {key}", flush=True)
    header = ["model", "total", "active", "KV/token", "s", "KV at s", "GFLOPs/token", "attention share"]
    if args.hub:
        header += ["exact (meta device)", "formula error"]
    rep.table(header, rows)
    if args.hub:
        rep("Exact counts come from the transformers model classes built from each repo's config.json. They include")
        rep("norm gains and biases, which the formulas skip. DeepSeek-V3's class leaves out the multi-token-prediction")
        rep("module, so its count is the main model only.")
    print(f"saved {rep.save()}")


if __name__ == "__main__":
    main()
