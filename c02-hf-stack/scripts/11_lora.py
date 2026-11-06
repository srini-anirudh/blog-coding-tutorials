"""Section 18: LoRA on SmolLM2-135M-Instruct. Inspect it, check the math, train it, merge it.

Run: python c02-hf-stack/scripts/11_lora.py [--steps 150] [--device auto]
"""

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, PrinterCallback, Trainer, TrainingArguments

from hfstack.chat import assistant_labels_by_prefix, chat_ids
from hfstack.collators import CausalLMCollator
from hfstack.data import make_arithmetic_rows, question, SYSTEM_PROMPT, to_messages
from hfstack.generation import generate_with_cache
from hfstack.hub import list_files
from hfstack.lora import add_lora, adapter_size_bytes, lora_shapes, manual_lora_forward, merge_matches
from hfstack.utils import CHAT_LM, be_polite, count_parameters, pick_device

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=150)
parser.add_argument("--device", default="auto")
args = parser.parse_args()
be_polite()
device = pick_device() if args.device == "auto" else torch.device(args.device)


@torch.no_grad()
def accuracy(model, tok, rows: list[dict]) -> float:
    """Exact-match accuracy of greedy answers, batched with left padding."""
    tok.padding_side = "left"
    correct = 0
    for start in range(0, len(rows), 32):
        chunk = rows[start : start + 32]
        prompts = [
            chat_ids(
                tok,
                [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question(r)}],
                add_generation_prompt=True,
            )
            for r in chunk
        ]
        enc = tok.pad({"input_ids": prompts}, return_tensors="pt").to(device)
        out = generate_with_cache(
            model, enc["input_ids"], enc["attention_mask"], 6,
            eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id,
        )
        replies = tok.batch_decode(out[:, enc["input_ids"].size(1) :], skip_special_tokens=True)
        correct += sum(reply.strip() == str(r["answer"]) for reply, r in zip(replies, chunk))
    return correct / len(rows)


tok = AutoTokenizer.from_pretrained(CHAT_LM)
base = AutoModelForCausalLM.from_pretrained(CHAT_LM, dtype=torch.float32).to(device)
test_rows = make_arithmetic_rows(1200, seed=0)[1000:]
train_rows = make_arithmetic_rows(1200, seed=0)[:1000]
before = accuracy(base, tok, test_rows)

model = add_lora(base, r=8, alpha=16)
trainable, total = count_parameters(model)
print(f"device {device}")
print(f"trainable {trainable:,} of {total:,} ({100 * trainable / total:.2f}%)")
for name, shape in lora_shapes(model)[:4]:
    print(f"  {name:<78} {shape}")
print(f"  ... {len(lora_shapes(model))} trainable tensors in total")

# The math by hand, on one wrapped layer. Give B nonzero weights first: at init it is zero.
layer = model.base_model.model.model.layers[0].self_attn.q_proj
print("\nwrapped layer:", type(layer).__module__ + "." + type(layer).__name__)
print("lora_B starts at zero:", layer.lora_B["default"].weight.abs().max().item() == 0.0)
with torch.no_grad():
    x = torch.randn(2, 5, layer.in_features, device=device)
    saved = layer.lora_B["default"].weight.clone()
    layer.lora_B["default"].weight.normal_(std=0.01)
    print(f"|peft - by hand| max: {(layer(x) - manual_lora_forward(layer, x)).abs().max().item():.2e}, scale = {layer.scaling['default']}")
    layer.lora_B["default"].weight.copy_(saved)

features = [assistant_labels_by_prefix(tok, to_messages(r)["messages"]) for r in train_rows]
tok.padding_side = "right"
with tempfile.TemporaryDirectory() as tmp:
    training = TrainingArguments(
        output_dir=tmp,
        per_device_train_batch_size=16,
        max_steps=args.steps,
        learning_rate=2e-3,
        warmup_steps=0.05,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        disable_tqdm=True,
        use_cpu=device.type == "cpu",
        seed=0,
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training,
        train_dataset=features,
        data_collator=CausalLMCollator(pad_token_id=tok.pad_token_id),
    )
    trainer.remove_callback(PrinterCallback)
    start = time.perf_counter()
    trainer.train()
    seconds = time.perf_counter() - start
curve = [(e["step"], e["loss"]) for e in trainer.state.log_history if "loss" in e and "train_loss" not in e]
print(f"\ntrained {args.steps} steps in {seconds:.0f} s ({seconds / args.steps:.2f} s/step)")
print("loss:", [(s, round(v, 3)) for s, v in curve[:: max(1, len(curve) // 8)]])

model.eval()
after = accuracy(model, tok, test_rows)
print(f"\nexact-match accuracy on {len(test_rows)} held-out sums: before {before:.3f}, after {after:.3f}")

with tempfile.TemporaryDirectory() as tmp:
    size = adapter_size_bytes(model, tmp)
    print("\nadapter checkpoint:")
    for name, bytes_ in list_files(tmp):
        print(f"  {name:<28} {bytes_:>12,}")
    print(f"full model in float32 would be {4 * total:,} bytes; the adapter is {100 * size / (4 * total):.2f}% of that")
    print("adapter_config target_modules:", sorted(json.loads((Path(tmp) / 'adapter_config.json').read_text())["target_modules"]))

    # Reload: base from the Hub, adapter from disk. Tokens must match the in-memory adapter.
    ids = torch.tensor([chat_ids(tok, to_messages(test_rows[0])["messages"][:2], add_generation_prompt=True)], device=device)
    reference = model(input_ids=ids).logits
    fresh = AutoModelForCausalLM.from_pretrained(CHAT_LM, dtype=torch.float32).to(device)
    reloaded = PeftModel.from_pretrained(fresh, tmp).eval()
    print(f"|logits after reload - before| max: {(reloaded(input_ids=ids).logits - reference).abs().max().item():.2e}")

merged, gap = merge_matches(model, ids)
print(f"\nmerge_and_unload: type {type(merged).__name__}, |logits merged - adapter| max: {gap:.2e}")
print("q_proj after merge:", type(merged.model.layers[0].self_attn.q_proj).__name__)
