"""One small experiment per tutorial step. Every output in the post comes from here.

    python c04-llm/scripts/walkthrough.py --step all
    python c04-llm/scripts/walkthrough.py --step rope
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tinyllm import ByteTokenizer, TinyLLM, get_config, load_checkpoint  # noqa: E402
from tinyllm.attention_naive import causal_mask, naive_attention, repeat_kv, split_heads  # noqa: E402
from tinyllm.attention_sdpa import sdpa_attention  # noqa: E402
from tinyllm.loss import causal_lm_loss  # noqa: E402
from tinyllm.mlp import SwiGLU  # noqa: E402
from tinyllm.norm import RMSNorm  # noqa: E402
from tinyllm.rope import RotaryEmbedding, rotate  # noqa: E402
from tinyllm.sampling import apply_top_k, apply_top_p  # noqa: E402

TOK = ByteTokenizer()


def rms(x: torch.Tensor) -> float:
    return x.float().pow(2).mean().sqrt().item()


def step_config() -> None:
    c = get_config("tiny")
    print(c)
    print(f"head_dim = {c.hidden_size} // {c.num_heads} = {c.head_dim}")
    print(f"query heads per kv head = {c.num_heads} // {c.num_kv_heads} = {c.num_kv_groups}")
    try:
        get_config("tiny", num_kv_heads=3)
    except ValueError as e:
        print(f"num_kv_heads=3 -> ValueError: {e}")


def step_embeddings() -> None:
    torch.manual_seed(0)
    model = TinyLLM(get_config("tiny"))
    ids = torch.tensor([TOK.encode("The cat sat")])
    x = model.embed_tokens(ids)
    print(f"ids {ids.shape}: {ids[0].tolist()}")
    print(f"x   {x.shape}  rms={rms(x):.4f}")
    print(f"the two 'a' bytes (positions 5 and 9) get the same vector: {torch.equal(x[0, 5], x[0, 9])}")
    print(f"tied: lm_head.weight is embed_tokens.weight -> {model.lm_head.weight is model.embed_tokens.weight}")
    untied = TinyLLM(get_config("tiny", tie_word_embeddings=False))
    print(f"params tied {model.num_parameters():,}  untied {untied.num_parameters():,}  "
          f"difference {untied.num_parameters() - model.num_parameters():,} = V*D = 256*256")


def step_rmsnorm() -> None:
    torch.manual_seed(0)
    norm = RMSNorm(256, eps=1e-5)
    for scale in (100.0, 1.0, 0.02, 0.001):
        x = torch.randn(1, 4, 256) * scale
        print(f"input rms {rms(x):9.4f} -> output rms {rms(norm(x)):.4f}")
    x = torch.randn(1, 4, 256) + 3.0
    print(f"input mean per token {x.mean(-1)[0, 0].item():.3f}")
    print(f"  RMSNorm keeps the sign of the mean: {norm(x).mean(-1)[0, 0].item():.3f}")
    print(f"  LayerNorm removes it:               {torch.nn.LayerNorm(256)(x).mean(-1)[0, 0].item():.3f}")
    ref = torch.nn.RMSNorm(256, eps=1e-5)
    print(f"max |ours - torch.nn.RMSNorm| = {(norm(x) - ref(x)).abs().max().item():.1e}")


def step_rope() -> None:
    torch.manual_seed(0)
    rope = RotaryEmbedding(head_dim=32, max_seq_len=256)
    print(f"inv_freq {rope.inv_freq.shape}, first three {rope.inv_freq[:3].tolist()}, last {rope.inv_freq[-1].item():.6f}")
    print(f"cos table {rope.cos_cached.shape}")
    print(f"wavelength of fastest pair {2 * math.pi / rope.inv_freq[0].item():.1f} tokens, "
          f"slowest {2 * math.pi / rope.inv_freq[-1].item():,.0f} tokens")
    q, k = torch.randn(1, 1, 1, 32), torch.randn(1, 1, 1, 32)

    def score(m: int, n: int) -> float:
        cq, sq = rope(torch.tensor([[m]]))
        ck, sk = rope(torch.tensor([[n]]))
        return (rotate(q, cq, sq) * rotate(k, ck, sk)).sum().item()

    print(f"norm before {q.norm().item():.4f}, after rotating to position 100 "
          f"{rotate(q, *rope(torch.tensor([[100]]))).norm().item():.4f}")
    for m, n in [(5, 2), (103, 100), (205, 202), (5, 3)]:
        print(f"score(q at {m:3d}, k at {n:3d}), offset {m - n}: {score(m, n):+.4f}")


def step_attention() -> None:
    torch.manual_seed(0)
    model = TinyLLM(get_config("tiny")).eval()
    attn = model.layers[0].self_attn
    x = torch.randn(1, 11, 256)
    q = split_heads(attn.q_proj(x), attn.num_heads)
    k = split_heads(attn.k_proj(x), attn.num_kv_heads)
    v = split_heads(attn.v_proj(x), attn.num_kv_heads)
    print(f"q_proj {attn.q_proj.weight.shape}  k_proj {attn.k_proj.weight.shape}  v_proj {attn.v_proj.weight.shape}")
    print(f"q {q.shape}\nk {k.shape}\nv {v.shape}")
    print(f"repeat_kv(k, {attn.n_rep}) {repeat_kv(k, attn.n_rep).shape}")
    print(f"causal_mask(4).int()      {causal_mask(4).int().tolist()}")
    print(f"causal_mask(2, 5).int()   {causal_mask(2, 5).int().tolist()}")
    a, b = naive_attention(q, k, v, attn.n_rep), sdpa_attention(q, k, v, attn.n_rep)
    print(f"output {a.shape}, max |naive - sdpa| = {(a - b).abs().max().item():.2e}")


def step_swiglu() -> None:
    torch.manual_seed(0)
    mlp = SwiGLU(256, 704)
    x = torch.randn(1, 11, 256)
    gate, up = F.silu(mlp.gate_proj(x)), mlp.up_proj(x)
    print(f"x {x.shape} -> gate {gate.shape}, up {up.shape} -> product {(gate * up).shape} -> out {mlp(x).shape}")
    print(f"gate range [{gate.min().item():.3f}, {gate.max().item():.3f}]; silu never goes below -0.278")
    print(f"SwiGLU params: 3*D*F = {3 * 256 * 704:,}")
    print(f"classic MLP with F = 4D: 2*D*4D = {2 * 256 * 1024:,}")
    print(f"F that matches it, 8D/3 = {8 * 256 / 3:.1f}, rounded up to a multiple of 64 = 704")


def step_block() -> None:
    torch.manual_seed(0)
    model = TinyLLM(get_config("tiny")).eval()
    block = model.layers[0]
    x = torch.randn(1, 11, 256)  # unit scale, so the branch sizes are easy to compare
    cos, sin = model.rotary(torch.arange(11)[None])
    a = block.self_attn(block.input_layernorm(x), cos, sin)
    y = x + a
    m = block.mlp(block.post_attention_layernorm(y))
    print(f"input              {x.shape} rms={rms(x):.3f}")
    print(f"attention branch   {a.shape} rms={rms(a):.3f}")
    print(f"mlp branch         {m.shape} rms={rms(m):.3f}")
    print(f"output             {(y + m).shape} rms={rms(y + m):.3f}")
    out, aux = block(x, cos, sin)
    print(f"block(x) matches by hand: {torch.equal(out, y + m)}; aux loss for a dense block: {aux}")


def step_stack() -> None:
    torch.manual_seed(0)
    model = TinyLLM(get_config("tiny")).eval()
    ids = torch.tensor([TOK.encode("The cat sat")])
    out = model(ids)
    print(f"logits {out.logits.shape}  rms={rms(out.logits):.3f}")
    probs = torch.softmax(out.logits[0, -1], -1)
    print(f"next-token probabilities {probs.shape}, sum={probs.sum().item():.4f}, largest={probs.max().item():.4f} "
          f"(uniform would be {1 / 256:.4f})")
    changed = ids.clone()
    changed[0, 8:] = 0
    same = torch.equal(model(changed).logits[:, :8], out.logits[:, :8])
    print(f"changing tokens 8.. leaves logits 0..7 unchanged: {same}")


def step_loss() -> None:
    torch.manual_seed(0)
    model = TinyLLM(get_config("tiny")).eval()
    ids = torch.tensor([TOK.encode("The cat")])
    print(f"input  {[chr(i) for i in ids[0, :-1].tolist()]}")
    print(f"target {[chr(i) for i in ids[0, 1:].tolist()]}")
    logits = model(ids).logits
    log_probs = torch.log_softmax(logits[0, :-1], dim=-1)
    manual = -log_probs[torch.arange(6), ids[0, 1:]].mean()
    print(f"by hand: mean -log p(next) = {manual.item():.4f}")
    print(f"causal_lm_loss             = {causal_lm_loss(logits, ids).item():.4f}")
    print(f"ln V                       = {math.log(256):.4f}")
    labels = ids.clone()
    labels[0, :4] = -100  # the prompt, as in a chat template
    print(f"with labels[:4] = -100, only {(labels[0, 1:] != -100).sum().item()} positions count: "
          f"{causal_lm_loss(logits, labels).item():.4f}")


def step_sampling() -> None:
    ckpt = ROOT / "runs" / "tiny-shakespeare"
    if not (ckpt / "model.safetensors").exists():
        print("run scripts/train_sanity.py first: this step needs a trained checkpoint")
        return
    model = load_checkpoint(ckpt).eval()
    prompt = "ROMEO:\nI will "
    ids = torch.tensor([TOK.encode(prompt)])
    logits = model(ids).logits[0, -1]
    # The post's interactive sampling figure replays these exact logits.
    dump = {"prompt": prompt, "logits": [round(x, 3) for x in logits.tolist()]}
    (ROOT / "results" / "sampling_logits.json").write_text(json.dumps(dump) + "\n")
    probs = torch.softmax(logits, -1)
    top = torch.topk(probs, 12)
    print(f"prompt {prompt!r}; top bytes for the next token")
    print("  " + ", ".join(f"{chr(i)!r} {p:.4f}" for p, i in zip(top.values.tolist(), top.indices.tolist())))
    print(f"  mass outside these 12: {1 - top.values.sum().item():.4f}")
    for k in (40, 5):
        print(f"top_k={k:<3} keeps {torch.isfinite(apply_top_k(logits[None], k)).sum().item()} tokens")
    for p in (0.95, 0.5):
        print(f"top_p={p:<4} keeps {torch.isfinite(apply_top_p(logits[None], p)).sum().item()} tokens")
    for t in (0.5, 1.0, 2.0):
        q = torch.softmax(logits / t, -1)
        h = -(q * q.clamp_min(1e-12).log()).sum().item()
        print(f"temperature {t}: top prob {q.max().item():.3f}, entropy {h:.2f} nats")


STEPS = {
    "config": step_config, "embeddings": step_embeddings, "rmsnorm": step_rmsnorm, "rope": step_rope,
    "attention": step_attention, "swiglu": step_swiglu, "block": step_block, "stack": step_stack,
    "loss": step_loss, "sampling": step_sampling,
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--step", default="all", choices=["all", *STEPS])
    args = p.parse_args()
    with torch.no_grad():
        for name, fn in STEPS.items():
            if args.step in ("all", name):
                print(f"--- {name} ---")
                fn()


if __name__ == "__main__":
    main()
