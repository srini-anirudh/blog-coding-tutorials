"""A local Hugging Face model as an agent backend.

Tool-calling instruct models (Qwen2.5, Hermes, SmolLM3, Llama 3.1+) ship a chat template
that renders the tool schemas into the system prompt and teaches the model to answer
with <tool_call>{...}</tool_call>. We let the template do the rendering, generate
greedily, and parse the text with the same parser every client uses.
"""

from __future__ import annotations

import time

from ..messages import ModelResponse, Usage
from ..parsing import response_from_text


def pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class HFLocalModel:
    def __init__(self, model_id: str = "Qwen/Qwen2.5-0.5B-Instruct", device: str | None = None,
                 max_new_tokens: int = 384, dtype: str = "float32"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = model_id
        self.device = device or pick_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=getattr(torch, dtype))
        self.model.to(self.device).eval()
        self.max_new_tokens = max_new_tokens
        self.last_prompt = ""      # the exact rendered prompt, for inspection

    def render(self, messages: list[dict], tools: list[dict]) -> str:
        """The prompt string the model actually sees, tool schemas included."""
        return self.tokenizer.apply_chat_template(messages, tools=tools or None,
                                                  add_generation_prompt=True, tokenize=False)

    def complete(self, messages: list[dict], tools: list[dict]) -> ModelResponse:
        import torch

        self.last_prompt = self.render(messages, tools)
        ids = self.tokenizer(self.last_prompt, return_tensors="pt").input_ids.to(self.device)
        # ids: (1, T_prompt) token ids
        t0 = time.perf_counter()
        with torch.no_grad():
            out = self.model.generate(ids, max_new_tokens=self.max_new_tokens, do_sample=False,
                                      pad_token_id=self.tokenizer.eos_token_id)
        # out: (1, T_prompt + T_new); the reply is the tail
        new = out[0, ids.shape[1]:]
        # <tool_call> is a plain added token in Qwen's vocab, but skip_special_tokens would also
        # drop it in some tokenizers, so decode everything and strip the end markers ourselves.
        text = self.tokenizer.decode(new, skip_special_tokens=False)
        for marker in ("<|im_end|>", "<|endoftext|>", "<|eot_id|>"):
            text = text.split(marker)[0]
        resp = response_from_text(text.strip(), Usage(ids.shape[1], int(new.shape[0])))
        resp.raw = text
        self.last_latency_s = time.perf_counter() - t0
        return resp
