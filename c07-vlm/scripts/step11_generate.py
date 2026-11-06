"""Step 11: generation. Hand-written greedy decoding against lm.generate, and the cost of no cache."""

import _common  # noqa: F401
import random
import time

import torch

from tinyvlm import build_tinyvlm
from tinyvlm.data import render
from tinyvlm.generation import greedy_generate

model, processor = build_tinyvlm()
image = render(2, "red", "circle", random.Random(0))
prompt = processor.apply_chat_template(
    [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What do you see?"}]}], add_generation_prompt=True
)
batch = processor(prompt, [image])
args = (batch["input_ids"], batch["pixel_values"], batch["attention_mask"])
print("prompt tokens", batch["input_ids"].shape[1], "(untrained projector)")

start = time.perf_counter()
hf = model.generate(*args, max_new_tokens=20, do_sample=False, eos_token_id=None, pad_token_id=2)
t_hf = time.perf_counter() - start
start = time.perf_counter()
cached = greedy_generate(model, *args, max_new_tokens=20)
t_cached = time.perf_counter() - start
start = time.perf_counter()
recomputed = greedy_generate(model, *args, max_new_tokens=20, use_cache=False)
t_recomputed = time.perf_counter() - start

print("output:", repr(processor.batch_decode(hf)[0]))
print("manual == lm.generate:", torch.equal(hf, cached), "| cached == recomputed:", torch.equal(cached, recomputed))
print(f"lm.generate        {t_hf:6.2f} s for 20 tokens (CPU, 2 threads)")
print(f"manual with cache  {t_cached:6.2f} s")
print(f"manual no cache    {t_recomputed:6.2f} s")
