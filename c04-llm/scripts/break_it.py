"""Break the model on purpose, one bug at a time, and measure the symptom each bug produces.

Each case patches one function for the duration of the experiment; the library code stays correct.

    python c04-llm/scripts/break_it.py --case all
    python c04-llm/scripts/break_it.py --case rope-layout --pretrained HuggingFaceTB/SmolLM2-135M
"""

from __future__ import annotations

import argparse
import sys
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tinyllm.attention as attention_mod  # noqa: E402
from tinyllm import ByteTokenizer, KVCache, TinyLLM, get_config, load_checkpoint  # noqa: E402
from tinyllm.utils import pick_device  # noqa: E402


@contextmanager
def patched(obj, name: str, value):
    old = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, old)


# ---------------------------------------------------------------------------------------------
def buggy_sdpa(q, k, v, n_rep):
    """The bug: is_causal=True whenever more than one query arrives. With a cache, SDPA aligns the triangle top-left."""
    return F.scaled_dot_product_attention(q, k, v, is_causal=q.shape[2] > 1, enable_gqa=n_rep > 1)


@torch.no_grad()
def case_is_causal_with_cache() -> None:
    torch.manual_seed(0)
    model = TinyLLM(get_config("tiny")).eval()
    ids = torch.randint(0, 256, (1, 24))
    full = model(ids).logits

    def chunked() -> torch.Tensor:
        cache = KVCache()
        return torch.cat([model(ids[:, s : s + 8], past_key_values=cache).logits for s in range(0, 24, 8)], 1)

    good = (chunked() - full).abs().max().item()
    with patched(attention_mod, "BACKENDS", {**attention_mod.BACKENDS, "sdpa": buggy_sdpa}):
        bad = (chunked() - full).abs().amax(dim=(0, 2))
    print("[is-causal-with-cache] prefill in 3 chunks of 8 on top of a cache, vs one full forward")
    print(f"  correct shifted mask: max |diff| = {good:.2e}")
    print(f"  is_causal=True:       max |diff| = {bad.max().item():.2e}")
    print(f"  per-position max diff (first chunk is fine, later chunks are not): "
          f"{[round(x, 2) for x in bad.tolist()]}")


# ---------------------------------------------------------------------------------------------
@torch.no_grad()
def case_position_offset() -> None:
    """The bug: decode steps forget the cache offset, so every new token is placed at position 0."""
    ckpt = ROOT / "runs" / "tiny-shakespeare"
    if not (ckpt / "model.safetensors").exists():
        print("[position-offset] run scripts/train_sanity.py first: random weights hide this bug")
        return
    model = load_checkpoint(ckpt).eval()
    tok = ByteTokenizer()
    prompt = torch.tensor([tok.encode("ROMEO:\nI will")])
    ref = model.generate(prompt, max_new_tokens=60, use_cache=False)
    with patched(KVCache, "get_seq_length", lambda self, layer_idx=0: 0):
        bad = model.generate(prompt, max_new_tokens=60, use_cache=True)
    diverge = (ref != bad)[0].nonzero()
    first = diverge[0].item() - prompt.shape[1] if len(diverge) else None
    print("[position-offset] trained tiny model, cached decode with positions restarting at 0")
    print(f"  identical to uncached generation: {torch.equal(ref, bad)}; first wrong new token: {first}")
    print(f"  correct: {tok.decode(ref[0].tolist())!r}")
    print(f"  buggy:   {tok.decode(bad[0].tolist())!r}")


# ---------------------------------------------------------------------------------------------
def apply_rotary_interleaved(q, k, cos, sin):
    """The bug: GPT-J / original-Meta layout, pairs (0,1), (2,3), ... instead of (i, i + d/2)."""
    half = q.shape[-1] // 2
    cos = cos[..., :half].repeat_interleave(2, dim=-1)[:, None].to(q.dtype)
    sin = sin[..., :half].repeat_interleave(2, dim=-1)[:, None].to(q.dtype)

    def rot(x):
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return x * cos + torch.stack((-x2, x1), dim=-1).flatten(-2) * sin

    return rot(q), rot(k)


@torch.no_grad()
def case_rope_layout(pretrained: str | None) -> None:
    print("[rope-layout] same weights, RoPE pairs channels (2i, 2i+1) instead of (i, i + d/2)")
    if pretrained is None:
        print("  (pass --pretrained HuggingFaceTB/SmolLM2-135M to see the effect on a trained model)")
        return
    from transformers import AutoTokenizer

    from tinyllm.hf import from_pretrained

    tok = AutoTokenizer.from_pretrained(pretrained)
    model = from_pretrained(pretrained)
    text = ("The quick brown fox jumps over the lazy dog. Paris is the capital of France, "
            "and Berlin is the capital of Germany.")
    ids = tok(text, return_tensors="pt").input_ids
    good = model(ids, labels=ids).loss.item()
    prompt = tok("The capital of France is", return_tensors="pt").input_ids
    good_txt = tok.decode(model.generate(prompt, max_new_tokens=12)[0])
    with patched(attention_mod, "apply_rotary", apply_rotary_interleaved):
        bad = model(ids, labels=ids).loss.item()
        bad_txt = tok.decode(model.generate(prompt, max_new_tokens=12)[0])
    print(f"  loss on a plain English sentence: correct {good:.3f} nats/token, wrong layout {bad:.3f}")
    print(f"  correct:      {good_txt!r}")
    print(f"  wrong layout: {bad_txt!r}")


# ---------------------------------------------------------------------------------------------
def case_no_shift(device: torch.device) -> None:
    """The bug: score logits[t] against labels[t]. The model learns to copy its input."""
    torch.manual_seed(0)
    tok = ByteTokenizer()
    path = ROOT / "data" / "tinyshakespeare.txt"
    if not path.exists():
        print("  run scripts/train_sanity.py first: it downloads the corpus")
        return
    text = path.read_text()
    data = torch.tensor(tok.encode(text))
    model = TinyLLM(get_config("tiny", num_layers=2)).to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    for step in range(1, 101):
        starts = torch.randint(0, data.numel() - 128, (16,))
        x = torch.stack([data[s : s + 128] for s in starts.tolist()]).to(device)
        logits = model(x).logits
        loss = F.cross_entropy(logits.reshape(-1, 256), x.reshape(-1))  # no shift
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step in (1, 25, 50, 100):
            print(f"  step {step:3d}  unshifted loss {loss.item():.4f}")
    out = model.generate(torch.tensor([tok.encode("ROMEO:")], device=device), max_new_tokens=40)
    print(f"  sample: {tok.decode(out[0].tolist())!r}")


# ---------------------------------------------------------------------------------------------
@torch.no_grad()
def case_init() -> None:
    print("[init] initial loss on random tokens (V = 256, ln V = 5.545)")
    ids = torch.randint(0, 256, (4, 128), generator=torch.Generator().manual_seed(0))
    for std in (0.02, 0.2, 1.0):
        torch.manual_seed(0)
        model = TinyLLM(get_config("tiny", initializer_range=std)).eval()
        print(f"  std={std:<5} loss {model(ids, labels=ids).loss.item():8.3f}")

    print("[init] residual-stream RMS at the last layer of a 48-layer model, std 0.02, embeddings RMS 0.02")
    for scaled in (True, False):
        torch.manual_seed(0)
        config = get_config("tiny", num_layers=48)
        model = TinyLLM(config).eval()
        if not scaled:  # the bug: o_proj and down_proj use the full std, not std / sqrt(2L)
            for name, module in model.named_modules():
                if name.endswith(("o_proj", "down_proj")):
                    torch.nn.init.normal_(module.weight, std=config.initializer_range)
        x = model.embed_tokens(ids)
        cos, sin = model.rotary(torch.arange(128).expand(4, 128))
        rms = []
        for layer in model.layers:
            x, _ = layer(x, cos, sin)
            rms.append(x.pow(2).mean().sqrt().item())
        label = "std/sqrt(2L) on residual projections" if scaled else "same std everywhere            "
        print(f"  {label}: layer 1 {rms[0]:.3f}, layer 12 {rms[11]:.3f}, layer 48 {rms[-1]:.3f}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--case", default="all",
                   choices=["all", "is-causal-with-cache", "position-offset", "rope-layout", "no-shift", "init"])
    p.add_argument("--pretrained", default=None)
    p.add_argument("--device", default=None)
    args = p.parse_args()
    device = pick_device(args.device)
    run = lambda c: args.case in ("all", c)  # noqa: E731
    if run("is-causal-with-cache"):
        case_is_causal_with_cache()
    if run("position-offset"):
        case_position_offset()
    if run("rope-layout"):
        case_rope_layout(args.pretrained)
    if run("no-shift"):
        print(f"[no-shift] 100 training steps with an unshifted loss, device={device}")
        case_no_shift(device)
    if run("init"):
        case_init()


if __name__ == "__main__":
    main()
