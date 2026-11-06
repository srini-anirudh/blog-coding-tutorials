"""Chat templates: from a list of messages to token IDs and assistant-only labels.

Two ways to find which tokens the assistant wrote:

1. Prefix diffing. Render the conversation up to each assistant turn and see which new tokens appear.
   Works with any template that is prefix-preserving (adding a message never changes earlier text).
2. Generation tags. A training template marks assistant text with `{% generation %}`, and
   `apply_chat_template(..., return_assistant_tokens_mask=True)` returns the mask directly.

Both must give the same labels; `tests/test_chat.py` checks that they do.
"""

from __future__ import annotations

from transformers import PreTrainedTokenizerBase

IGNORE_INDEX = -100  # PyTorch cross_entropy skips targets equal to this value

Message = dict[str, str]


def chat_ids(tokenizer: PreTrainedTokenizerBase, messages: list[Message], add_generation_prompt: bool = False) -> list[int]:
    """Render `messages` with the tokenizer's chat template and return token IDs as a flat list."""
    # transformers 5 returns a BatchEncoding when tokenize=True; ask for the plain dict and take input_ids.
    out = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=add_generation_prompt, return_dict=True
    )
    return list(out["input_ids"])


def assistant_labels_by_prefix(tokenizer: PreTrainedTokenizerBase, messages: list[Message]) -> dict[str, list[int]]:
    """Tokenize a conversation and keep labels only on assistant tokens.

    Returns `input_ids`, `attention_mask` and `labels`, each of length N (one conversation, no padding).
    Labels are -100 everywhere except assistant content and its end-of-turn token.
    """
    input_ids = chat_ids(tokenizer, messages)
    labels = [IGNORE_INDEX] * len(input_ids)
    end_of_turn = tokenizer.eos_token_id

    for i, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        # Tokens before this reply: everything up to and including the "<|im_start|>assistant\n" header.
        prefix = chat_ids(tokenizer, messages[:i], add_generation_prompt=True)
        through = chat_ids(tokenizer, messages[: i + 1])
        if input_ids[: len(through)] != through or through[: len(prefix)] != prefix:
            raise ValueError("chat template is not prefix-preserving; use a template with {% generation %} tags")
        start, stop = len(prefix), len(through)
        # Train on the reply and its end-of-turn token, not on the newline the template adds after it.
        while stop > start and through[stop - 1] != end_of_turn:
            stop -= 1
        labels[start:stop] = input_ids[start:stop]

    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}


# SmolLM2's template with `{% generation %}` markers around what the assistant writes.
# The rendered text is identical to the original template; only the markers are new.
SMOLLM2_TRAINING_TEMPLATE = (
    "{% for message in messages %}"
    "{% if loop.first and messages[0]['role'] != 'system' %}"
    "{{ '<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n' }}"
    "{% endif %}"
    "{% if message['role'] == 'assistant' %}"
    "{{ '<|im_start|>assistant\n' }}"
    "{% generation %}{{ message['content'] + '<|im_end|>' }}{% endgeneration %}"
    "{{ '\n' }}"
    "{% else %}"
    "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)


def assistant_labels_by_template(
    tokenizer: PreTrainedTokenizerBase, messages: list[Message], chat_template: str = SMOLLM2_TRAINING_TEMPLATE
) -> dict[str, list[int]]:
    """Same output as `assistant_labels_by_prefix`, using the template's generation markers instead."""
    out = tokenizer.apply_chat_template(
        messages,
        chat_template=chat_template,
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
    )
    input_ids = list(out["input_ids"])
    mask = list(out["assistant_masks"])
    labels = [tok if m else IGNORE_INDEX for tok, m in zip(input_ids, mask)]
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}
