"""Thinking control as string surgery on the decode loop (section "Budgets and mode switching").

Every control here edits the token stream. None touches a weight.

  hard off   prefill an empty, closed think block; the model is past the point where it could think
  soft off   append "/no_think" to the user turn; the model was trained to obey, which is not a guarantee
  budget     Qwen3: count thinking tokens, and at the cap splice in the stop-thinking sentence and </think>,
             then let the model answer from the partial trace ("deliberate until interrupted, then salvage")
  extend     s1: when the model emits </think>, delete it and append "Wait", so it keeps going

The controller is written against a `generate` callable so it can be tested without a model:

  generate(prompts: list[list[int]], max_new_tokens: int, stop_ids: set[int]) -> list[list[int]]

returns each prompt's new tokens, ending with the stop id if one was produced. `thinktok.hf` provides one
backed by a Hugging Face model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

Generate = Callable[[list[list[int]], int, set[int]], list[list[int]]]

# Verbatim from the Qwen3 technical report, section on thinking budgets.
QWEN3_STOP_THINKING = (
    "Considering the limited time by the user, I have to give the solution based on the thinking directly now.\n"
    "</think>.\n\n"
)
S1_WAIT = "Wait"


class Tokenizer(Protocol):
    def encode(self, text: str, add_special_tokens: bool = ...) -> list[int]: ...
    def decode(self, ids: list[int], skip_special_tokens: bool = ...) -> str: ...


@dataclass
class ThinkTokens:
    think_end: int  # id of </think>
    eos: int  # id that ends the assistant turn (<|im_end|> for Qwen3)


@dataclass
class Trace:
    prompt: list[int]
    stream: list[int] = field(default_factory=list)  # everything after the prompt, in order
    thinking: list[int] = field(default_factory=list)  # tokens the model wrote while thinking
    injected: list[int] = field(default_factory=list)  # tokens the controller spliced into the stream
    answer: list[int] = field(default_factory=list)  # tokens the model wrote after thinking
    truncated: bool = False  # the budget cut thinking off
    waits: int = 0  # times "Wait" replaced </think>

    def append(self, ids: list[int], kind: str) -> None:
        self.stream += ids
        getattr(self, kind).extend(ids)


def _answer_phase(generate: Generate, traces: list[Trace], toks: ThinkTokens, max_answer: int) -> None:
    open_ = [t for t in traces if not (t.stream and t.stream[-1] == toks.eos)]
    if not open_:
        return
    outs = generate([t.prompt + t.stream for t in open_], max_answer, {toks.eos})
    for t, o in zip(open_, outs):
        t.append(o, "answer")


def think_with_budget(
    generate: Generate, prompts: list[list[int]], tok: Tokenizer, toks: ThinkTokens, budget: int,
    max_answer: int = 512, stop_text: str = QWEN3_STOP_THINKING,
) -> list[Trace]:
    """Qwen3-style budget: think for at most `budget` tokens, then interrupt and salvage.

    budget == 0 means the hard off switch: prefill the closed empty think block.
    """
    traces = [Trace(prompt=list(p)) for p in prompts]
    if budget == 0:
        for t in traces:
            t.append(tok.encode("<think>\n\n</think>\n\n", add_special_tokens=False), "injected")
    else:
        outs = generate([t.prompt for t in traces], budget, {toks.think_end, toks.eos})
        stop_ids = tok.encode(stop_text, add_special_tokens=False)
        for t, o in zip(traces, outs):
            t.append(o, "thinking")
            if not o or o[-1] not in (toks.think_end, toks.eos):
                t.truncated = True
                t.append(stop_ids, "injected")
    _answer_phase(generate, traces, toks, max_answer)
    return traces


def think_with_waits(
    generate: Generate, prompts: list[list[int]], tok: Tokenizer, toks: ThinkTokens, waits: int,
    cap: int, max_answer: int = 512, wait_text: str = S1_WAIT,
) -> list[Trace]:
    """s1 budget forcing. Suppress </think> up to `waits` times by replacing it with "Wait".

    Thinking is capped at `cap` model-written tokens in total. At the cap the end-of-thinking delimiter is
    appended, as s1 does to cap, and the model answers.
    """
    traces = [Trace(prompt=list(p)) for p in prompts]
    wait_ids = tok.encode(wait_text, add_special_tokens=False)
    end_ids = tok.encode("\n</think>\n\n", add_special_tokens=False)
    active = list(traces)
    while active:
        rooms = [cap - len(t.thinking) for t in active]
        outs = generate([t.prompt + t.stream for t in active], max(rooms), {toks.think_end, toks.eos})
        still = []
        for t, o, room in zip(active, outs, rooms):
            o = o[:room]  # one batch call uses the largest room; each trace keeps only its own share
            stopped = bool(o) and o[-1] in (toks.think_end, toks.eos)
            if stopped and o[-1] == toks.think_end and t.waits < waits:
                t.append(o[:-1], "thinking")
                t.append(wait_ids, "injected")
                t.waits += 1
            else:
                t.append(o, "thinking")
                if stopped:
                    continue  # the model closed its thinking and no Wait is left
            if len(t.thinking) >= cap:
                t.truncated = True
                t.append(end_ids, "injected")
                continue
            still.append(t)
        active = still
    _answer_phase(generate, traces, toks, max_answer)
    return traces


def chat_prompt(tokenizer, question: str, mode: str) -> list[int]:
    """Token ids of a Qwen3 chat prompt. mode: 'think', 'soft_no_think', 'soft_think' or 'hard_no_think'."""
    content = question
    if mode == "soft_no_think":
        content += " /no_think"
    elif mode == "soft_think":
        content += " /think"
    elif mode not in ("think", "hard_no_think"):
        raise ValueError(f"unknown mode {mode!r}")
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True,
        enable_thinking=(mode != "hard_no_think"),
    )
    return tokenizer.encode(text, add_special_tokens=False)


def thought_was_empty(completion_text: str) -> bool:
    """True when the generated turn opens with an empty think block (the model chose not to think)."""
    s = completion_text.lstrip()
    if not s.startswith("<think>"):
        return True  # no think block at all also means no thinking
    return s[len("<think>"):].lstrip().startswith("</think>")
