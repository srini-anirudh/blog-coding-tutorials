"""Four small checks that back claims in the tutorial. Each one prints its evidence.

    python c04-llm/scripts/review_checks.py is-causal-single-query
    python c04-llm/scripts/review_checks.py position-offset-random
    python c04-llm/scripts/review_checks.py pytest-position-offset-bug
    python c04-llm/scripts/review_checks.py sdpa-cpu-backend

The outputs are saved in results/ under the same names, with a first line giving the hardware.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tinyllm.attention as attention_mod  # noqa: E402
from tinyllm import KVCache, TinyLLM, get_config  # noqa: E402


def zero_seq_length(self, layer_idx: int = 0) -> int:
    """The position-offset bug: the cache never reports how many tokens it holds."""
    return 0


@torch.no_grad()
def is_causal_single_query() -> None:
    """is_causal=True aligns the triangle to the top-left corner, so one query sees only key 0."""
    torch.manual_seed(0)
    q, k, v = torch.randn(1, 4, 1, 16), torch.randn(1, 4, 9, 16), torch.randn(1, 4, 9, 16)
    wrong = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    right = F.scaled_dot_product_attention(q, k, v)
    print(f"random q (1 query), k, v (9 keys): is_causal=True vs no mask: "
          f"max diff {(wrong - right).abs().max().item():.4f} | "
          f"equals value of key 0 only: {torch.allclose(wrong, v[:, :, :1].expand_as(wrong))}")

    # The literal bug: every call, including single-token decode, passes is_causal=True.
    def always_causal(q, k, v, n_rep):
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=n_rep > 1)

    def spy(q, k, v, n_rep):
        """Record, at every real decode call, how far is_causal=True is from the correct no-mask answer."""
        if q.shape[2] == 1 and k.shape[2] > 1:
            bad = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=n_rep > 1)
            good = F.scaled_dot_product_attention(q, k, v, enable_gqa=n_rep > 1)
            first = v[:, :, :1].repeat_interleave(n_rep, dim=1)
            seen.append(((bad - good).abs().max().item(), torch.allclose(bad, first)))
        return saved(q, k, v, n_rep)

    torch.manual_seed(0)  # same setup as scripts/break_it.py
    model = TinyLLM(get_config("tiny")).eval()
    ids = torch.randint(0, 256, (1, 24))
    full = model(ids).logits
    saved, seen = attention_mod.BACKENDS["sdpa"], []
    attention_mod.BACKENDS["sdpa"] = spy
    try:
        cache = KVCache()
        for t in range(24):
            model(ids[:, t : t + 1], past_key_values=cache)
        attention_mod.BACKENDS["sdpa"] = always_causal
        cache = KVCache()
        steps = torch.cat([model(ids[:, t : t + 1], past_key_values=cache).logits for t in range(24)], dim=1)
    finally:
        attention_mod.BACKENDS["sdpa"] = saved
    print(f"real decode calls of the tiny model: {len(seen)}, largest attention-output difference "
          f"{max(d for d, _ in seen):.4f}, every one equals key 0 only: {all(same for _, same in seen)}")
    print(f"token-by-token decode with literally always is_causal=True: "
          f"max |diff| vs full = {(steps - full).abs().max().item():.4f}")


@torch.no_grad()
def position_offset_random() -> None:
    """With random weights the position-offset bug changes nothing, which is why a trained model is needed."""
    saved = KVCache.get_seq_length
    for seed in range(5):
        torch.manual_seed(seed)
        model = TinyLLM(get_config("tiny")).eval()
        prompt = torch.randint(0, 256, (1, 12), generator=torch.Generator().manual_seed(seed))
        ref = model.generate(prompt, max_new_tokens=60, use_cache=False)
        KVCache.get_seq_length = zero_seq_length
        try:
            bad = model.generate(prompt, max_new_tokens=60, use_cache=True)
        finally:
            KVCache.get_seq_length = saved
        print(f"{seed} identical: {torch.equal(ref, bad)} mismatches: {(ref != bad).sum().item()}")


def pytest_position_offset_bug() -> None:
    """Run the offline tests with the position-offset bug patched in. Which ones notice?"""
    import pytest

    class Patch:
        def pytest_configure(self, config):
            KVCache.get_seq_length = zero_seq_length

    sys.exit(pytest.main(
        [str(ROOT), "-m", "not slow and not gpu and not network", "--tb=no", "-rf", "-p", "no:cacheprovider"],
        plugins=[Patch()],
    ))


def sdpa_cpu_backend() -> None:
    """Which kernel does F.scaled_dot_product_attention use on a CPU, and does it accept grouped heads?"""
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torch.profiler import ProfilerActivity, profile

    print(f"torch {torch.__version__}")
    q = torch.randn(1, 8, 16, 32, requires_grad=True)
    for gqa in (True, False):
        heads = 2 if gqa else 8
        k, v = torch.randn(1, heads, 16, 32), torch.randn(1, heads, 16, 32)
        for backend in (SDPBackend.FLASH_ATTENTION, SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION):
            try:
                with sdpa_kernel(backend):
                    out = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=gqa)
                print(f"{backend.name} enable_gqa {gqa} fp32 OK {type(out.grad_fn).__name__}")
            except RuntimeError as e:
                print(f"{backend.name} enable_gqa {gqa} fp32 FAIL {str(e).splitlines()[0]}")
    for gqa in (True, False):
        k = torch.randn(1, 2 if gqa else 8, 16, 32)
        out = F.scaled_dot_product_attention(q, k, k, is_causal=True, enable_gqa=gqa)
        print(f"default dispatch {'gqa' if gqa else 'no-gqa'} grad_fn: {type(out.grad_fn).__name__}")

    def ops(q_len: int) -> list[str]:
        qq, k = torch.randn(1, 8, q_len, 32), torch.randn(1, 2, 16, 32)
        with profile(activities=[ProfilerActivity.CPU]) as prof:
            F.scaled_dot_product_attention(qq, k, k, enable_gqa=True)
        return sorted(e.key for e in prof.key_averages())

    print("ops with enable_gqa on CPU:", ops(16))
    print("decode q_len=1 ops:", ops(1))


CHECKS = {
    "is-causal-single-query": is_causal_single_query,
    "position-offset-random": position_offset_random,
    "pytest-position-offset-bug": pytest_position_offset_bug,
    "sdpa-cpu-backend": sdpa_cpu_backend,
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("check", choices=CHECKS)
    CHECKS[p.parse_args().check]()


if __name__ == "__main__":
    main()
