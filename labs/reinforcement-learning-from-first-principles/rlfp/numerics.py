"""Section 14, point 6: two correct implementations of the same policy disagree in the last few bits.

The sampler (an inference engine) decodes one token at a time with a KV cache. The trainer runs one teacher-forced
forward over the whole sequence. Same weights, same tokens, different kernels and reduction orders. Here both paths
are plain PyTorch, in three dtypes, so the gap can be measured token by token.
"""

from __future__ import annotations

import math

import torch


@torch.no_grad()
def forward_logprobs(model, seqs: torch.Tensor, prompt_len: int) -> torch.Tensor:
    """Trainer path: one forward over prompt + response. log pi of each response token: (B, T), float32."""
    logits = model(input_ids=seqs).logits[:, prompt_len - 1:-1].float()
    return logits.log_softmax(-1).gather(-1, seqs[:, prompt_len:].unsqueeze(-1)).squeeze(-1)


@torch.no_grad()
def decode_logprobs(model, seqs: torch.Tensor, prompt_len: int) -> torch.Tensor:
    """Sampler path: prefill the prompt, then feed one token at a time through the KV cache: (B, T), float32."""
    out = model(input_ids=seqs[:, :prompt_len], use_cache=True)
    pkv, logits = out.past_key_values, out.logits[:, -1]
    lps = []
    for t in range(prompt_len, seqs.shape[1]):
        tok = seqs[:, t]
        lps.append(logits.float().log_softmax(-1).gather(-1, tok.unsqueeze(-1)).squeeze(-1))
        if t + 1 < seqs.shape[1]:
            out = model(input_ids=tok.unsqueeze(1), past_key_values=pkv, use_cache=True)
            pkv, logits = out.past_key_values, out.logits[:, -1]
    return torch.stack(lps, 1)


def mismatch_stats(lp_a: torch.Tensor, lp_b: torch.Tensor, lengths=(16, 64, 256), clip: float = 0.2) -> dict:
    """Per-token and per-sequence statistics of the log-ratio log pi_a - log pi_b over response tokens."""
    d = (lp_a - lp_b).double()
    sigma = float(d.std())
    out = {"token_sigma": sigma, "token_max_abs": float(d.abs().max()),
           "token_share_ratio_outside_clip": float(((d.exp() - 1).abs() > clip).double().mean())}
    for T in lengths:
        if T > d.shape[1]:
            continue
        s = d[:, :T].sum(1)
        out[f"seq_std_T{T}"] = float(s.std())
        out[f"sigma_sqrtT_T{T}"] = sigma * math.sqrt(T)
        out[f"seq_ratio_max_T{T}"] = float(s.exp().max())
        out[f"seq_ratio_min_T{T}"] = float(s.exp().min())
    return out
