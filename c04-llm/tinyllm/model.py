"""TinyLLM: embeddings -> L decoder blocks -> final RMSNorm -> LM head."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .block import DecoderBlock
from .config import ModelConfig
from .init import init_weights
from .kv_cache import KVCache
from .loss import causal_lm_loss
from .norm import RMSNorm
from .rope import RotaryEmbedding


@dataclass
class CausalLMOutput:
    logits: torch.Tensor  # (B, T, V)
    loss: torch.Tensor | None = None  # scalar; includes the MoE auxiliary loss when present
    past_key_values: KVCache | None = None
    aux_loss: torch.Tensor | None = None  # MoE load-balancing loss, summed over layers


class TinyLLM(nn.Module):
    """A Llama-architecture decoder. Attribute names match HF LlamaModel so weights map by name."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.rotary = RotaryEmbedding(config.head_dim, config.max_seq_len, config.rope_theta)
        self.layers = nn.ModuleList(DecoderBlock(config, i) for i in range(config.num_layers))
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            # One (V, D) matrix reads tokens in and scores tokens out.
            self.lm_head.weight = self.embed_tokens.weight
        init_weights(self, config)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        past_key_values: KVCache | None = None,
        use_cache: bool = False,
    ) -> CausalLMOutput:
        """input_ids: (B, T) token ids. With a cache, only the new tokens are passed in."""
        b, t = input_ids.shape
        if use_cache and past_key_values is None:
            past_key_values = KVCache()
        past = past_key_values.get_seq_length() if past_key_values is not None else 0
        if past + t > self.config.max_seq_len:
            raise ValueError(f"sequence length {past + t} exceeds max_seq_len={self.config.max_seq_len}")

        x = self.embed_tokens(input_ids)  # (B, T, D)
        # New tokens sit after the cached ones, so their positions start at `past`.
        position_ids = torch.arange(past, past + t, device=input_ids.device).expand(b, t)
        cos, sin = self.rotary(position_ids)  # (B, T, head_dim) each, shared by all layers

        aux_total = None
        for layer in self.layers:
            x, aux = layer(x, cos, sin, past_key_values)
            if aux is not None:
                aux_total = aux if aux_total is None else aux_total + aux

        logits = self.lm_head(self.norm(x))  # (B, T, V)

        loss = None
        if labels is not None:
            loss = causal_lm_loss(logits, labels)
            if aux_total is not None:
                loss = loss + self.config.router_aux_loss_coef * aux_total
        return CausalLMOutput(logits=logits, loss=loss, past_key_values=past_key_values, aux_loss=aux_total)

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        """See tinyllm.generation.generate for the arguments."""
        from .generation import generate

        return generate(self, input_ids, **kwargs)

    def num_parameters(self) -> int:
        """Unique parameters. A tied weight is counted once (parameters() already deduplicates)."""
        return sum(p.numel() for p in self.parameters())
