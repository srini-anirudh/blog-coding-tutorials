"""Sections 20-21: what save_pretrained writes, the safetensors layout, shards and the index file.

push_to_hub appears as code only. This script never uploads anything.

Run: python c02-hf-stack/scripts/13_hub_and_safetensors.py
"""

import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from hfstack.hub import list_files, read_index, read_safetensors_header, read_tensor_by_hand
from hfstack.utils import CHAT_LM, be_polite

be_polite()
tok = AutoTokenizer.from_pretrained(CHAT_LM)
model = AutoModelForCausalLM.from_pretrained(CHAT_LM, dtype=torch.float32)

PUSH_EXAMPLE = """
# Not executed here. After `huggingface-cli login`:
model.push_to_hub("your-name/smollm2-arithmetic")
tok.push_to_hub("your-name/smollm2-arithmetic")
"""


def checksum(m) -> float:
    return sum(p.double().sum().item() for p in m.parameters())


with tempfile.TemporaryDirectory() as tmp:
    single = Path(tmp) / "single"
    model.save_pretrained(single)
    tok.save_pretrained(single)
    print("--- save_pretrained + tokenizer.save_pretrained ---")
    for name, size in list_files(single):
        print(f"  {name:<28} {size:>12,}")
    print("\ngeneration_config.json:", json.loads((single / "generation_config.json").read_text()))

    print("\n--- inside model.safetensors, parsed by hand ---")
    weights = single / "model.safetensors"
    print("first 8 bytes as little-endian u64 = header length:", int.from_bytes(weights.read_bytes()[:8], "little"))
    header = read_safetensors_header(weights)
    print("metadata:", header.pop("__metadata__", None))
    print("tensors :", len(header))
    for name in list(header)[:3]:
        print(f"  {name:<46} {header[name]}")
    print("lm_head.weight in the file:", "lm_head.weight" in header, "(tied, so saved once)")
    name = "model.layers.0.self_attn.q_proj.weight"
    by_hand = read_tensor_by_hand(weights, name)
    print("tensor read with seek() equals the model's:", torch.equal(by_hand, model.state_dict()[name]))

    print("\n--- sharded: max_shard_size='200MB' ---")
    sharded = Path(tmp) / "sharded"
    model.save_pretrained(sharded, max_shard_size="200MB")
    for name, size in list_files(sharded):
        print(f"  {name:<40} {size:>12,}")
    index = read_index(sharded)
    print("index metadata:", index["metadata"])
    entries = list(index["weight_map"].items())
    print("weight_map, first and last:", entries[0], entries[-1])
    per_shard: dict[str, int] = {}
    for _, shard in entries:
        per_shard[shard] = per_shard.get(shard, 0) + 1
    print("tensors per shard:", per_shard)

    start = time.perf_counter()
    reloaded = AutoModelForCausalLM.from_pretrained(sharded, dtype=torch.float32)
    sharded_seconds = time.perf_counter() - start
    print(f"reload from shards in {sharded_seconds:.2f} s; identical weights: {checksum(model) == checksum(reloaded)}")

    print("\n--- only the config: no weights touched ---")
    start = time.perf_counter()
    config = AutoConfig.from_pretrained(single)
    print(f"config in {1000 * (time.perf_counter() - start):.1f} ms, hidden_size {config.hidden_size}")
    start = time.perf_counter()
    with torch.device("meta"):
        skeleton = AutoModelForCausalLM.from_config(config)
    print(f"meta-device skeleton in {1000 * (time.perf_counter() - start):.0f} ms, {sum(p.numel() for p in skeleton.parameters()):,} parameters, 0 bytes allocated")

print("\npush_to_hub (shown, not executed):" + PUSH_EXAMPLE)
