"""Chat template and assistant-only labels.

The base SmolLM2-135M tokenizer ships without a chat template, but its vocabulary already contains the ChatML
markers `<|im_start|>` (id 1) and `<|im_end|>` (id 2). We attach a ChatML template and build labels by hand,
one message at a time, so every token knows which role produced it.
"""

from __future__ import annotations

from dataclasses import dataclass

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from .utils import BASE_MODEL

IGNORE_INDEX = -100

# ChatML, as used by SmolLM2-Instruct, without its default system prompt. The {% generation %} markers let
# Hugging Face compute an assistant mask too, which the tests compare against our manual mask.
CHAT_TEMPLATE = (
    "{%- for message in messages %}"
    "{{- '<|im_start|>' + message['role'] + '\\n' }}"
    "{%- if message['role'] == 'assistant' %}"
    "{% generation %}{{- message['content'] + '<|im_end|>' }}{% endgeneration %}{{- '\\n' }}"
    "{%- else %}"
    "{{- message['content'] + '<|im_end|>' + '\\n' }}"
    "{%- endif %}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- endif %}"
)


def load_tokenizer(name: str = BASE_MODEL) -> PreTrainedTokenizerBase:
    """Base tokenizer + ChatML template. Pad is <|endoftext|>, never <|im_end|> (see `encode_chat`)."""
    tok = AutoTokenizer.from_pretrained(name)
    tok.chat_template = CHAT_TEMPLATE
    tok.pad_token = "<|endoftext|>"
    tok.padding_side = "right"
    return tok


@dataclass
class ChatExample:
    input_ids: list[int]
    labels: list[int]
    roles: list[str]  # which message produced each token: system/user/assistant, plus "header"/"sep"


def encode_chat(tok: PreTrainedTokenizerBase, messages: list[dict[str, str]], mode: str = "assistant") -> ChatExample:
    """Tokenize a conversation and build its labels.

    mode="assistant": train on every assistant reply (content + <|im_end|>), mask everything else.
    mode="last":      train on the final assistant reply only.
    mode="all":       train on every token, the way pretraining would.

    The "<|im_start|>assistant\\n" header is masked because at inference we feed it ourselves
    (add_generation_prompt). The closing <|im_end|> is a target: it is how the model learns to stop.
    """
    ids: list[int] = []
    labels: list[int] = []
    roles: list[str] = []
    last_assistant = max(i for i, m in enumerate(messages) if m["role"] == "assistant")

    def add(text: str, role: str, train: bool) -> None:
        piece = tok(text, add_special_tokens=False)["input_ids"]
        ids.extend(piece)
        labels.extend(piece if (train or mode == "all") else [IGNORE_INDEX] * len(piece))
        roles.extend([role] * len(piece))

    for i, m in enumerate(messages):
        is_target = m["role"] == "assistant" and (mode != "last" or i == last_assistant)
        add(f"<|im_start|>{m['role']}\n", "header", train=False)
        add(m["content"] + "<|im_end|>", m["role"], train=is_target)
        add("\n", "sep", train=False)
    return ChatExample(ids, labels, roles)


def encode_prompt(tok: PreTrainedTokenizerBase, messages: list[dict[str, str]]) -> list[int]:
    """Token ids of a prompt ending in the assistant header, ready for generation."""
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tok(text, add_special_tokens=False)["input_ids"]


def mask_table(tok: PreTrainedTokenizerBase, ex: ChatExample) -> list[dict]:
    """One row per token, for printing or for the interactive loss-mask figure."""
    rows = []
    for pos, (tid, lab, role) in enumerate(zip(ex.input_ids, ex.labels, ex.roles)):
        rows.append({"pos": pos, "id": tid, "text": tok.decode([tid]), "role": role, "label": lab})
    return rows
