"""Section 14, point 6: your on-policy algorithm is not on-policy.

Sample completions from SmolLM2-135M, then score the same tokens with the same weights along two paths: the
sampler path (token-by-token decode with a KV cache) and the trainer path (one teacher-forced forward). Do it in
float32, bfloat16 and float16. The per-token log-ratio is tiny; its sum over a sequence spreads like sigma*sqrt(T),
and the sequence-level ratio is its exponential.

This is the mechanism at small scale, with plain PyTorch on both sides. The papers measured vLLM or SGLang against
FSDP or Megatron on large models; that was not reproduced here.

    python labs/reinforcement-learning-from-first-principles/scripts/10_on_policy_illusion.py            # SmolLM2-135M
    python labs/reinforcement-learning-from-first-principles/scripts/10_on_policy_illusion.py --random   # offline
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rlfp.numerics import decode_logprobs, forward_logprobs, mismatch_stats  # noqa: E402
from rlfp.report import Report  # noqa: E402

PROMPTS = ["The capital of France is", "def fibonacci(n):", "Photosynthesis is the process by which",
           "In 1905, Albert Einstein"]


def load(random: bool, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig, LlamaForCausalLM

    if random:
        torch.manual_seed(0)
        cfg = LlamaConfig(vocab_size=512, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                          num_attention_heads=4, num_key_value_heads=2)
        return LlamaForCausalLM(cfg).to(device).eval(), None
    name = "HuggingFaceTB/SmolLM2-135M"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32).to(device).eval()
    return model, tok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--random", action="store_true", help="tiny random Llama, no download")
    ap.add_argument("--samples", type=int, default=4, help="completions per prompt")
    ap.add_argument("--new-tokens", type=int, default=256)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rep = Report(args.name or ("10_on_policy_illusion" + ("_random" if args.random else "")
                               + ("_gpu" if device == "cuda" else "")))
    model, tok = load(args.random, device)
    if tok is None:
        prompt_ids = torch.randint(0, 512, (len(PROMPTS), 6), generator=torch.Generator().manual_seed(0))
    else:
        enc = [tok(p).input_ids for p in PROMPTS]
        n = min(len(e) for e in enc)  # equal-length prompts, so no padding is needed
        prompt_ids = torch.tensor([e[:n] for e in enc])
    prompt_ids = prompt_ids.repeat_interleave(args.samples, 0).to(device)
    Lp = prompt_ids.shape[1]
    torch.manual_seed(0)
    with torch.no_grad():
        seqs = model.generate(prompt_ids, attention_mask=torch.ones_like(prompt_ids), do_sample=True, top_k=0,
                              top_p=1.0, temperature=1.0, max_new_tokens=args.new_tokens,
                              min_new_tokens=args.new_tokens, pad_token_id=0)
    rep.print(f"model: {'random tiny Llama' if args.random else 'SmolLM2-135M'}, {seqs.shape[0]} completions x "
              f"{seqs.shape[1] - Lp} tokens, sampled in float32, device {device}")

    lp = {}
    for dtype in (torch.float32, torch.bfloat16, torch.float16):
        m = model.to(dtype)
        name = str(dtype).replace("torch.", "")
        lp[f"forward_{name}"] = forward_logprobs(m, seqs, Lp)
        lp[f"decode_{name}"] = decode_logprobs(m, seqs, Lp)
    model.to(torch.float32)

    pairs = [("decode_float32", "forward_float32", "sampler vs trainer, both float32"),
             ("decode_bfloat16", "forward_bfloat16", "sampler vs trainer, both bfloat16"),
             ("decode_float16", "forward_float16", "sampler vs trainer, both float16"),
             ("forward_bfloat16", "forward_float32", "trainer bfloat16 vs float32"),
             ("forward_float16", "forward_float32", "trainer float16 vs float32")]
    rows = []
    for a, b, label in pairs:
        st = mismatch_stats(lp[a], lp[b], lengths=(16, 64, seqs.shape[1] - Lp))
        rows.append({"comparison": label, **st})
    T = seqs.shape[1] - Lp
    cols = ["comparison", "token_sigma", "token_max_abs", "token_share_ratio_outside_clip", "seq_std_T16",
            "seq_std_T64", f"seq_std_T{T}", f"sigma_sqrtT_T{T}", f"seq_ratio_min_T{T}", f"seq_ratio_max_T{T}"]
    rep.table(rows, cols=cols, fmt="{:.3g}")
    rep.print("token_sigma: std of log pi_a - log pi_b per token. seq_std_TN: std of that log-ratio summed over the "
              "first N tokens. token_share_ratio_outside_clip: share of tokens with |ratio - 1| > 0.2.")
    rep.data["rows"] = rows
    rep.save()


if __name__ == "__main__":
    main()
