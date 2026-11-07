"""Chat templates, the assistant-only loss mask, and what truncation does to it.

A conversation is a list of {"role", "content"} dicts. The tokenizer's chat template flattens it
to text. We find each assistant turn by rendering prefixes of the conversation (this works for any
template that only appends), then map character spans to tokens with the tokenizer's offsets.

An `Encoded` example holds token ids and one loss weight per token. Weight w[t] is the weight of
the term -log p(x_t | x_<t) in the loss. Token 0 is never predicted, so its weight is irrelevant.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

IM = re.compile(r"<\|im_start\|>(\w+)\n(.*?)<\|im_end\|>", re.S)  # ChatML turns


def render(tok, messages, add_generation_prompt: bool = False) -> str:
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)


def assistant_spans(tok, messages) -> list[tuple[int, int]]:
    """Character spans of every assistant turn: from the first content character to the end of the
    turn-end token (the token that follows is excluded). Raises if the template is not append-only."""
    full = render(tok, messages)
    spans = []
    for i, m in enumerate(messages):
        if m["role"] != "assistant":
            continue
        start = len(render(tok, messages[:i], add_generation_prompt=True))
        upto = render(tok, messages[: i + 1])
        if not full.startswith(upto):
            raise ValueError("template is not append-only; prefix rendering does not match")
        k = upto.find(tok.eos_token, start)
        end = k + len(tok.eos_token) if k >= 0 else len(upto)
        spans.append((start, end))
    return spans


def role_ranges(text: str):
    """ChatML only. Yields (role, header_start, content_start, content_end, turn_end)."""
    for m in IM.finditer(text):
        yield m.group(1), m.start(), m.start(2), m.end(2), m.end()


@dataclass
class Encoded:
    ids: list[int]
    w: list[float]                       # loss weight per token
    n_turns: int = 1                     # assistant turns in the conversation
    truncated: bool = False
    spans: list = field(default_factory=list)  # token index ranges [a, b) of assistant turns

    def __len__(self):
        return len(self.ids)

    @property
    def scored(self) -> int:
        return sum(1 for x in self.w[1:] if x > 0)


def encode_chat(tok, messages, mode: str = "assistant", max_len: int | None = None, prompt_weight: float = 0.3,
                truncate: str = "right") -> Encoded:
    """Tokenise one conversation and attach loss weights.

    mode "assistant": 1 on assistant content and its turn-end token, 0 elsewhere.
    mode "last":      the same, but only for the final assistant turn.
    mode "all":       1 on every token (the pretraining loss on a chat-formatted string).
    mode "weighted":  assistant tokens 1, user-turn content `prompt_weight`, role headers, system
                      prompt and turn-end tokens of other roles 0 (instruction modelling).
    max_len: cut the sequence at that many tokens (`truncate="right"` keeps the start).
    """
    text = render(tok, messages)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, off = enc["input_ids"], enc["offset_mapping"]
    spans = assistant_spans(tok, messages)
    w = [0.0] * len(ids)

    def inside(a, ranges):
        return any(s <= a[0] and a[1] <= e and a[0] < a[1] for s, e in ranges)

    tok_spans = []
    for s, e in spans:
        idx = [t for t, o in enumerate(off) if s <= o[0] and o[1] <= e and o[0] < o[1]]
        tok_spans.append((idx[0], idx[-1] + 1) if idx else (0, 0))
    use = spans[-1:] if mode == "last" else spans
    for t, o in enumerate(off):
        if inside(o, use):
            w[t] = 1.0
    if mode == "all":
        w = [1.0] * len(ids)
    elif mode == "weighted":
        user = [(cs, ce) for role, _, cs, ce, _ in role_ranges(text) if role == "user"]
        for t, o in enumerate(off):
            if w[t] == 0.0 and inside(o, user):
                w[t] = prompt_weight
    elif mode not in ("assistant", "last"):
        raise ValueError(mode)
    truncated = False
    if max_len is not None and len(ids) > max_len:
        ids, w, truncated = ids[:max_len], w[:max_len], True
        tok_spans = [(a, min(b, max_len)) for a, b in tok_spans if a < max_len]
    return Encoded(ids, w, len(spans), truncated, tok_spans)


def encode_prompt(tok, messages) -> list[int]:
    """Ids for generation: the conversation so far plus the assistant header, nothing else."""
    return tok(render(tok, messages, add_generation_prompt=True), add_special_tokens=False)["input_ids"]


def has_turn_end(enc: Encoded, tok) -> bool:
    """Is the final assistant turn's end token still inside the (possibly truncated) sequence, and scored?"""
    eot = tok.convert_tokens_to_ids(tok.eos_token)
    return any(i == eot and wt > 0 for i, wt in zip(enc.ids, enc.w))


def show_mask(tok, enc: Encoded) -> str:
    """One line per token: id, text, weight. Used by the printed examples."""
    rows = []
    for t, (i, wt) in enumerate(zip(enc.ids, enc.w)):
        rows.append(f"{t:4d} {i:7d} {tok.convert_ids_to_tokens(i)!r:<22} w={wt:g}")
    return "\n".join(rows)
