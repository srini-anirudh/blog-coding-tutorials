"""ChatML rendering, split into pieces that know whether they are supervised.

SmolLM2-Instruct uses ChatML:
    <|im_start|>user\\n{content}<|im_end|>\\n<|im_start|>assistant\\n{reply}<|im_end|>\\n

An image is one <image> marker inside a user message. The processor later expands each marker into
N_img placeholder tokens per crop.

One deliberate difference: SmolLM2's own template inserts a default system message when there is none.
We do not, so the sequence stays the one we are building: role header, image, question, assistant.
<|im_start|> is the tokenizer's BOS token, so every sequence still starts with BOS.
"""

from __future__ import annotations

IMAGE_TOKEN = "<image>"
IM_START, IM_END = "<|im_start|>", "<|im_end|>"


def render_content(content: str | list[dict]) -> str:
    """Content is a plain string or a list of parts: {"type": "image"} or {"type": "text", "text": ...}."""
    if isinstance(content, str):
        return content
    return "".join(IMAGE_TOKEN if part["type"] == "image" else part["text"] for part in content)


def chat_segments(messages: list[dict], add_generation_prompt: bool = False) -> list[tuple[str, bool]]:
    """Return (text, supervised) pieces whose concatenation is the full ChatML string.

    Only the assistant's reply and its closing <|im_end|> are supervised: the model must learn to
    answer and to stop. Role headers, the user's words and the image are context, not targets.
    """
    segments: list[tuple[str, bool]] = []
    for message in messages:
        segments.append((f"{IM_START}{message['role']}\n", False))
        body = render_content(message["content"])
        if message["role"] == "assistant":
            segments.append((body + IM_END, True))
            segments.append(("\n", False))
        else:
            segments.append((body + IM_END + "\n", False))
    if add_generation_prompt:
        segments.append((f"{IM_START}assistant\n", False))
    return segments


def apply_chat_template(messages: list[dict], add_generation_prompt: bool = False) -> str:
    return "".join(text for text, _ in chat_segments(messages, add_generation_prompt))
