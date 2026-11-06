"""The ledger: where fine-tuning memory goes, and which part LoRA removes (post: "The ledger nobody reads", Fig. 1, 4).

Part 1 is exact arithmetic for Llama-2-7B and GPT-3 175B geometry, checking the numbers the post quotes.
Part 2 measures the same ledger on a real model: a SmolLM2-135M-shaped Llama with random weights, one AdamW step,
counting the bytes of every gradient and optimizer-state tensor, and the size of the saved checkpoint.
"""

import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from loralab.ledger import (FULL_FT_BYTES, GB, GPT3_175B, LLAMA2_7B, TARGET_SETS, TRAINABLE_BYTES, break_even_rank,
                            lora_params, nf4_bits_per_param, training_memory_gb)
from loralab.lora import adapter_state_dict, apply_lora
from loralab.report import Report, table


def tensor_bytes(ts) -> int:
    return sum(t.numel() * t.element_size() for t in ts if t is not None)


def measure(model, lr=1e-3):
    """One AdamW step, then count what the optimizer machinery holds."""
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr)
    ids = torch.randint(0, model.config.vocab_size, (2, 64))
    model(input_ids=ids, labels=ids).loss.backward()
    opt.step()
    state = [t for s in opt.state.values() for t in s.values() if torch.is_tensor(t) and t.dim() > 0]
    return {"params_total": sum(p.numel() for p in model.parameters()),
            "params_trainable": sum(p.numel() for p in trainable),
            "weights_bytes": tensor_bytes(model.parameters()),
            "grad_bytes": tensor_bytes(p.grad for p in trainable),
            "adam_state_bytes": tensor_bytes(state)}


def checkpoint_bytes(tensors: dict) -> int:
    from safetensors.torch import save_file
    seen, unique = set(), {}
    for k, v in tensors.items():  # tied embeddings share storage; save them once, as transformers does
        if v.data_ptr() not in seen:
            seen.add(v.data_ptr())
            unique[k] = v.contiguous()
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "ckpt.safetensors"
        save_file(unique, str(path))
        return path.stat().st_size


def main():
    rep = Report("01_ledger")
    g = LLAMA2_7B
    n = g.total_params()
    rep("## Part 1: arithmetic, Llama-2-7B geometry (d=4096, 32 layers, SwiGLU 11008, vocab 32000)")
    rep(f"total parameters: {n / 1e9:.3f} B")
    rep("bytes per parameter, full fine-tuning:", ", ".join(f"{k} {v}" for k, v in FULL_FT_BYTES.items()),
        f"= {TRAINABLE_BYTES}")
    rep(f"7B x 16 B = {7e9 * 16 / GB:.0f} GB; the bf16 weights alone are {7e9 * 2 / GB:.0f} GB "
        f"({2 / 16:.3f} of the total)")
    attn = sum(o * i for k, (o, i) in g.matrices().items() if k in TARGET_SETS["All attention"])
    mlp = sum(o * i for k, (o, i) in g.matrices().items() if k not in TARGET_SETS["All attention"])
    rep(f"per layer: attention {attn / 1e6:.1f} M, MLP {mlp / 1e6:.1f} M, MLP share {mlp / (attn + mlp):.3f}")
    rep(f"square d x d, r=8: r(2d)/d^2 = {8 * 2 * 4096 / 4096 ** 2:.4%}")
    rep("break-even rank (adapter stores as many numbers as the matrix):",
        f"4096x4096 -> {break_even_rank(4096, 4096):.0f}, 11008x4096 -> {break_even_rank(11008, 4096):.0f}")

    rows = []
    for r in (1, 8, 16, 64, 256, 512):
        for name, targets in TARGET_SETS.items():
            t = lora_params(g, r, targets)
            mem = training_memory_gb(g, "lora", r, targets)
            rows.append([r, name, f"{t / 1e6:.2f} M", f"{t / n:.3%}", f"{mem['total']:.2f}"])
    rep("\nLoRA trainable parameters and training memory (frozen bf16 at 2 B, trainable at 16 B, no activations)")
    for line in table(rows, ["rank", "targets", "trainable", "share", "memory GB"]):
        rep(line)
    rep()
    for method in ("full", "lora", "qlora"):
        m = training_memory_gb(g, method, 16)
        rep(f"{method:>5} (r=16, all linear): frozen {m['frozen']:.2f} GB + trainable {m['trainable']:.2f} GB "
            f"= {m['total']:.2f} GB")
    bits_single, bits_double = nf4_bits_per_param(double_quant=False), nf4_bits_per_param()
    rep(f"NF4 bits/param: single quant {bits_single:.3f}, double quant {bits_double:.4f} "
        f"(saving {bits_single - bits_double:.3f} bits); 6.74 B at {bits_double:.3f} bits = "
        f"{n * bits_double / 8 / GB:.2f} GB vs bf16 {n * 2 / GB:.2f} GB")

    g3 = GPT3_175B
    qv4 = lora_params(g3, 4, ("q_proj", "v_proj"))
    q8 = lora_params(g3, 8, ("q_proj",))
    rep(f"\nGPT-3 175B (d=12288, 96 layers): Wq,Wv at r=4 -> {qv4 / 1e6:.2f} M params = {qv4 * 2 / 1e6:.1f} MB in fp16;"
        f" Wq alone at r=8 -> {q8 / 1e6:.2f} M (the 18 M budget of Table 5)")
    rep(f"GPT-3 weights in fp16: {175e9 * 2 / GB:.0f} GB, so the checkpoint shrinks by {175e9 * 2 / (qv4 * 2):.0f}x")
    rep.data["arithmetic"] = {"llama2_7b_params": n, "attn_per_layer": attn, "mlp_per_layer": mlp,
                              "nf4_bits_single": bits_single, "nf4_bits_double": bits_double,
                              "gpt3_qv_r4_params": qv4, "rows": rows}

    rep("\n## Part 2: measured, SmolLM2-135M shapes (576 wide, 30 layers, GQA 9/3 heads), random weights, fp32")
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(hidden_size=576, intermediate_size=1536, num_hidden_layers=30, num_attention_heads=9,
                      num_key_value_heads=3, vocab_size=49152, tie_word_embeddings=True, max_position_embeddings=2048)
    measured = {}
    for label, r in (("full fine-tuning", None), ("LoRA r=8, all linear", 8), ("LoRA r=64, all linear", 64)):
        torch.manual_seed(0)
        model = LlamaForCausalLM(cfg)
        if r:
            apply_lora(model, r=r, alpha=2 * r)
        m = measure(model)
        sd = model.state_dict() if r is None else adapter_state_dict(model)
        m["checkpoint_bytes"] = checkpoint_bytes(sd)
        measured[label] = m
        del model
    rows = []
    for label, m in measured.items():
        rows.append([label, f"{m['params_trainable'] / 1e6:.2f} M", f"{m['weights_bytes'] / 1e6:.0f}",
                     f"{m['grad_bytes'] / 1e6:.1f}", f"{m['adam_state_bytes'] / 1e6:.1f}",
                     f"{m['checkpoint_bytes'] / 1e6:.2f}"])
    for line in table(rows, ["setting", "trainable", "weights MB", "grads MB", "Adam m+v MB", "checkpoint MB"]):
        rep(line)
    full, lo = measured["full fine-tuning"], measured["LoRA r=8, all linear"]
    rep(f"grads + Adam state: full {(full['grad_bytes'] + full['adam_state_bytes']) / 1e6:.0f} MB, LoRA r=8 "
        f"{(lo['grad_bytes'] + lo['adam_state_bytes']) / 1e6:.1f} MB; both are exactly 12 B per trainable parameter "
        f"in fp32 ({(lo['grad_bytes'] + lo['adam_state_bytes']) / lo['params_trainable']:.1f})")
    rep(f"checkpoint: full {full['checkpoint_bytes'] / 1e6:.0f} MB vs LoRA r=8 {lo['checkpoint_bytes'] / 1e6:.2f} MB "
        f"({full['checkpoint_bytes'] / lo['checkpoint_bytes']:.0f}x smaller)")
    rep.data["measured_smollm2_shape"] = measured
    rep.save()


if __name__ == "__main__":
    main()
