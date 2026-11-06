"""A small causal LM wrapper for the two real-model experiments.

`greedy_matches` checks whether greedy decoding would produce `answer` without generating:
run the prompt once, then feed the answer tokens, and test that the argmax at every answer
position is the next answer token. That is exactly "greedy output starts with the answer"
(ties aside), and on a CPU it is much cheaper than generating token by token.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


@dataclass
class Check:
    correct: bool
    logprob: float  # log P(answer | prompt), summed over answer tokens
    prompt_tokens: int
    needle_attention: float | None = None  # attention share on the needle span, if asked


class LM:
    def __init__(self, name: str = DEFAULT_MODEL, device: str = "cpu", eager: bool = False):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(name)
        kw = {"attn_implementation": "eager"} if eager else {}
        self.model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32, **kw).to(device).eval()
        self.device = device
        self.name = name

    def prompt_ids(self, user: str, system: str | None = None) -> list[int]:
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return self.tok(text, add_special_tokens=False)["input_ids"]

    @torch.no_grad()
    def greedy_matches(self, user: str, answers: str | list[str], system: str | None = None,
                       needle: str | None = None) -> Check:
        """Would greedy decoding start with one of `answers` (for example the value with or without quotes)?"""
        answers = [answers] if isinstance(answers, str) else answers
        p = self.prompt_ids(user, system)
        span = self._span(user, needle, system) if needle else None
        grabbed: list[torch.Tensor] = []
        hooks = []
        if span is not None:
            # Keep only the attention row of the last prompt token, per layer, averaged over heads.
            def grab(_m, _inp, out):
                w = out[1]
                if w is not None:
                    grabbed.append(w[0, :, -1, :].float().mean(0).cpu())
            hooks = [layer.self_attn.register_forward_hook(grab) for layer in self.model.model.layers]
        try:
            out = self.model(torch.tensor([p], device=self.device), use_cache=True)
        finally:
            for h in hooks:
                h.remove()
        first = out.logits[0, -1].float()
        att = None
        if span is not None and grabbed:
            rows = torch.stack(grabbed)  # layers x seq
            att = float(rows[:, span[0]:span[1]].sum(-1).mean())
        best: tuple[bool, float] | None = None
        for ans in answers:
            a = self.tok(ans, add_special_tokens=False)["input_ids"]
            lp = float(torch.log_softmax(first, -1)[a[0]])
            ok = int(first.argmax()) == a[0]
            if len(a) > 1:
                cache = copy.deepcopy(out.past_key_values)
                rest = self.model(torch.tensor([a[:-1]], device=self.device), past_key_values=cache).logits[0].float()
                target = torch.tensor(a[1:], device=self.device)
                ok = ok and bool((rest.argmax(-1) == target).all())
                lp += float(torch.log_softmax(rest, -1).gather(1, target[:, None]).sum())
            if best is None or (ok, lp) > best:
                best = (ok, lp)
            if ok:
                break
        return Check(best[0], best[1], len(p), att)

    def _span(self, user: str, needle: str, system: str | None) -> tuple[int, int] | None:
        """Token range of `needle` inside the chat-formatted prompt, found through character offsets."""
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        start = text.find(needle)
        if start < 0:
            return None
        enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        toks = [i for i, (s, e) in enumerate(enc["offset_mapping"]) if e > start and s < start + len(needle)]
        return (toks[0], toks[-1] + 1) if toks else None
