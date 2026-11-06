"""A small Hugging Face chat model used as every agent in the real-model experiments.

One model plays all the roles. That is the honest setting for the post's question: the
agents differ only in what they are told and what they read.
"""

from __future__ import annotations

import time

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


class Agent:
    """Greedy or sampled chat completions, with a token counter for the bill."""

    def __init__(self, model_id: str = DEFAULT_MODEL, device: str = "cpu", dtype: str = "float32"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.name = model_id
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.tok.padding_side = "left"                # batched generation needs left padding
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=getattr(torch, dtype))
        self.model.to(device).eval()
        self.prompt_tokens = 0
        self.output_tokens = 0
        self.calls = 0
        self.seconds = 0.0

    def render(self, user: str, system: str | None = None) -> str:
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
        return self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)

    def n_tokens(self, text: str) -> int:
        return len(self.tok(text).input_ids)

    def ask(self, users: list[str], system: str | list[str] | None = None, max_new_tokens: int = 24,
            temperature: float = 0.0, seed: int | None = None) -> list[str]:
        """One batched call. `system` may be one string or one per prompt."""
        torch = self.torch
        systems = system if isinstance(system, list) else [system] * len(users)
        texts = [self.render(u, s) for u, s in zip(users, systems)]
        enc = self.tok(texts, return_tensors="pt", padding=True).to(self.device)
        # enc.input_ids: (B, T) left-padded prompt ids
        kw = dict(max_new_tokens=max_new_tokens, pad_token_id=self.tok.eos_token_id)
        if temperature > 0:
            kw.update(do_sample=True, temperature=temperature, top_p=1.0, top_k=0)
            if seed is not None:
                torch.manual_seed(seed)
        else:
            kw.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = self.model.generate(**enc, **kw)
        self.seconds += time.perf_counter() - t0
        new = out[:, enc.input_ids.shape[1]:]          # (B, T_new)
        self.prompt_tokens += int(enc.attention_mask.sum())
        self.output_tokens += int((new != self.tok.eos_token_id).sum())
        self.calls += len(users)
        return [self.tok.decode(row, skip_special_tokens=True).strip() for row in new]

    def bill(self) -> dict:
        return {"calls": self.calls, "prompt_tokens": self.prompt_tokens,
                "output_tokens": self.output_tokens, "seconds": round(self.seconds, 1)}

    def reset_bill(self) -> None:
        self.prompt_tokens = self.output_tokens = self.calls = 0
        self.seconds = 0.0
