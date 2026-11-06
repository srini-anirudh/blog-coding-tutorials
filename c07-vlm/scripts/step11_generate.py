"""Step 11: generation. Hand-written greedy decoding against lm.generate, and the cost of no cache."""

import _common  # noqa: F401
import random
import time

import torch

from tinyvlm import build_tinyvlm, describe_device
from tinyvlm.data import render
from tinyvlm.generation import greedy_generate


def best_of(fn, repeats: int = 5) -> tuple[torch.Tensor, float]:
    """The fastest of a few runs, because other work on the machine only ever adds time."""
    seconds = []
    for _ in range(repeats):
        start = time.perf_counter()
        output = fn()
        seconds.append(time.perf_counter() - start)
    return output, min(seconds)


model, processor = build_tinyvlm()
image = render(2, "red", "circle", random.Random(0))
prompt = processor.apply_chat_template(
    [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What do you see?"}]}], add_generation_prompt=True
)
batch = processor(prompt, [image])
args = (batch["input_ids"], batch["pixel_values"], batch["attention_mask"])
print("prompt tokens", batch["input_ids"].shape[1], "(untrained projector)")

hf, t_hf = best_of(lambda: model.generate(*args, max_new_tokens=20, do_sample=False, eos_token_id=None, pad_token_id=2))
cached, t_cached = best_of(lambda: greedy_generate(model, *args, max_new_tokens=20))
recomputed, t_recomputed = best_of(lambda: greedy_generate(model, *args, max_new_tokens=20, use_cache=False))

print("output:", repr(processor.batch_decode(hf)[0]))
print("manual == lm.generate:", torch.equal(hf, cached), "| cached == recomputed:", torch.equal(cached, recomputed))
print(f"20 new tokens, best of 5, {describe_device(batch['input_ids'].device)}")
print(f"lm.generate        {t_hf:6.2f} s  {20 / t_hf:6.1f} tokens/s")
print(f"manual with cache  {t_cached:6.2f} s  {20 / t_cached:6.1f} tokens/s")
print(f"manual no cache    {t_recomputed:6.2f} s  {20 / t_recomputed:6.1f} tokens/s")
