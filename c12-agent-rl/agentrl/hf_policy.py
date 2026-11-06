"""Drop-in adapters: run the same rollout and GRPO code with a Hugging Face causal LM.

``HFPolicy`` wraps any ``AutoModelForCausalLM`` and adds a value head, so it returns the same ``PolicyOutput`` as
``TinyPolicy``. ``HFCodec`` wraps the model's tokenizer. For the trajectory markers it reuses tokens the model
already knows (SmolLM2's chat delimiters), so LoRA alone can learn to end an action: a freshly added token would
need a trainable output row. Nothing in rollout.py, grpo.py or online.py changes when the policy is a pretrained LM.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .policy import PolicyOutput


class HFCodec:
    def __init__(self, tokenizer, act: str = "<|im_start|>", eos: str = "<|im_end|>", obs: str = "\n"):
        self.tok = tokenizer
        self.act, self.eos = tokenizer.convert_tokens_to_ids([act, eos])
        (self.obs,) = tokenizer(obs, add_special_tokens=False).input_ids
        self.pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        self.bos = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id

    def encode(self, text: str) -> list[int]:
        return self.tok(" " + text, add_special_tokens=False).input_ids

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=True).strip()


class HFPolicy(nn.Module):
    def __init__(self, model, vocab_size: int | None = None):
        super().__init__()
        if vocab_size is not None and vocab_size != model.get_input_embeddings().num_embeddings:
            model.resize_token_embeddings(vocab_size)
        self.model = model
        self.value_head = nn.Linear(model.config.hidden_size, 1)
        nn.init.zeros_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, input_ids: torch.Tensor) -> PolicyOutput:
        """Right-padded input needs no attention mask here either: causal attention never looks ahead."""
        out = self.model(input_ids=input_ids, output_hidden_states=True)
        h = out.hidden_states[-1]                                     # (B, T, D)
        return PolicyOutput(out.logits, self.value_head(h.to(self.value_head.weight.dtype)).squeeze(-1))


def tiny_llama_policy(vocab_size: int, hidden: int = 128, layers: int = 2, heads: int = 4) -> HFPolicy:
    """A randomly initialised LlamaForCausalLM the size of TinyPolicy: the trusted-implementation check."""
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(vocab_size=vocab_size, hidden_size=hidden, intermediate_size=4 * hidden,
                      num_hidden_layers=layers, num_attention_heads=heads, num_key_value_heads=heads,
                      max_position_embeddings=512, tie_word_embeddings=False)
    return HFPolicy(LlamaForCausalLM(cfg))


def smollm_policy(name: str = "HuggingFaceTB/SmolLM2-135M-Instruct", lora_rank: int = 16, device: str = "cpu"):
    """SmolLM2 with LoRA adapters on the attention projections, plus its codec."""
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    codec = HFCodec(AutoTokenizer.from_pretrained(name))
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32)
    model = get_peft_model(model, LoraConfig(r=lora_rank, lora_alpha=2 * lora_rank,
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    return HFPolicy(model).to(device), codec
