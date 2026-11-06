"""A custom architecture that plugs into the Hugging Face stack.

Three classes, the same split every model in `transformers` uses:

    BackwardPassConfig        architecture hyperparameters, serialised to config.json
    BackwardPassModel         embeddings → decoder blocks → final norm; returns hidden states
    BackwardPassForCausalLM   BackwardPassModel + LM head; returns logits and loss; can generate()

After `register()`, `AutoConfig`, `AutoModel` and `AutoModelForCausalLM` dispatch on
model_type="backward_pass", so save_pretrained / from_pretrained / Trainer / PEFT / generate all work.

The block is a small pre-norm decoder (RMSNorm, causal multi-head attention, GELU MLP, learned
positions). C3 and C4 build the modern attention and block; this file is about the integration.
"""

# No `from __future__ import annotations` here: transformers reads `config: BackwardPassConfig`
# at class-creation time and needs the real class, not a string.
import torch
import torch.nn.functional as F
from torch import nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    Cache,
    DynamicCache,
    GenerationMixin,
    PreTrainedConfig,
    PreTrainedModel,
)
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast


class BackwardPassConfig(PreTrainedConfig):
    # model_type is written into config.json; AutoConfig reads it back to pick this class.
    model_type = "backward_pass"
    keys_to_ignore_at_inference = ["past_key_values"]

    # In transformers 5 a config is a dataclass: annotated fields with defaults.
    vocab_size: int = 512
    hidden_size: int = 64
    num_hidden_layers: int = 2
    num_attention_heads: int = 4
    intermediate_size: int = 256
    max_position_embeddings: int = 256
    rms_norm_eps: float = 1e-6
    initializer_range: float = 0.02
    use_cache: bool = True
    tie_word_embeddings: bool = True
    pad_token_id: int | None = 0
    bos_token_id: int | None = 1
    eos_token_id: int | None = 2

    def validate_architecture(self):
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(f"hidden_size {self.hidden_size} is not divisible by {self.num_attention_heads} heads")


class BackwardPassRMSNorm(nn.Module):
    # The name contains "RMSNorm", so PreTrainedModel's default _init_weights sets the weight to ones.
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x32 = x.float()
        x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x32.to(x.dtype)


class BackwardPassAttention(nn.Module):
    def __init__(self, config: BackwardPassConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx  # which slot of the KV cache belongs to this layer
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.qkv_proj = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor, cache: Cache | None
    ) -> torch.Tensor:
        """x: (B, T, D). mask: (B, 1, T, S) boolean, True = may attend. Returns (B, T, D)."""
        B, T, D = x.shape
        q, k, v = self.qkv_proj(x).view(B, T, 3, self.num_heads, self.head_dim).unbind(dim=2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # each (B, H, T, d_h)
        if cache is not None:
            k, v = cache.update(k, v, self.layer_idx)  # (B, H, S, d_h), S = past + T
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)  # (B, H, T, d_h)
        return self.o_proj(out.transpose(1, 2).reshape(B, T, D))


class BackwardPassBlock(nn.Module):
    def __init__(self, config: BackwardPassConfig, layer_idx: int):
        super().__init__()
        self.attn_norm = BackwardPassRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attn = BackwardPassAttention(config, layer_idx)
        self.mlp_norm = BackwardPassRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.hidden_size),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor, cache: Cache | None) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), mask, cache)
        return x + self.mlp(self.mlp_norm(x))


class BackwardPassPreTrainedModel(PreTrainedModel):
    """Shared plumbing: which config class to use, the weight-name prefix, init, and capabilities."""

    config: BackwardPassConfig  # transformers 5 reads the config class from this annotation
    base_model_prefix = "model"  # weights of the base model live under "model." in the causal-LM class
    _no_split_modules = ["BackwardPassBlock"]  # device_map="auto" never splits a block across devices
    supports_gradient_checkpointing = True


def causal_padding_mask(attention_mask: torch.Tensor, query_len: int) -> torch.Tensor:
    """Combine causality and padding. attention_mask: (B, S) of 0/1 over all keys → (B, 1, T, S) bool.

    Query i (the last `query_len` positions) may attend to key j if j ≤ its absolute position and j is real.
    """
    S = attention_mask.size(1)
    q_pos = torch.arange(S - query_len, S, device=attention_mask.device)[:, None]  # (T, 1)
    k_pos = torch.arange(S, device=attention_mask.device)[None, :]  # (1, S)
    causal = k_pos <= q_pos  # (T, S)
    mask = causal[None, None] & attention_mask[:, None, None, :].bool()  # (B, 1, T, S)
    # A fully padded query row would softmax over nothing and give NaN; let it see itself instead.
    return mask | torch.eye(S, device=mask.device, dtype=torch.bool)[-query_len:][None, None]


class BackwardPassModel(BackwardPassPreTrainedModel):
    def __init__(self, config: BackwardPassConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.embed_positions = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.layers = nn.ModuleList(BackwardPassBlock(config, i) for i in range(config.num_hidden_layers))
        self.norm = BackwardPassRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_init()  # runs _init_weights on every submodule and sets up weight tying

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool | None = None,
        output_hidden_states: bool | None = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)  # (B, T, D)
        B, T, _ = inputs_embeds.shape
        use_cache = self.config.use_cache if use_cache is None else use_cache
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        past = past_key_values.get_seq_length() if past_key_values is not None else 0

        if attention_mask is None:  # (B, S) over past + current keys
            attention_mask = torch.ones(B, past + T, dtype=torch.long, device=inputs_embeds.device)
        if position_ids is None:  # skip left padding: count real tokens before each position
            position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)[:, -T:]

        mask = causal_padding_mask(attention_mask, T)
        h = inputs_embeds + self.embed_positions(position_ids)
        hidden_states = (h,) if output_hidden_states else None
        for layer in self.layers:
            h = layer(h, mask, past_key_values if use_cache else None)
            if output_hidden_states:
                hidden_states += (h,)
        h = self.norm(h)
        return BaseModelOutputWithPast(
            last_hidden_state=h,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=hidden_states,
        )


class BackwardPassForCausalLM(BackwardPassPreTrainedModel, GenerationMixin):
    # transformers 5 format: {tied parameter: source parameter}. Saved once, re-tied on load.
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: BackwardPassConfig):
        super().__init__(config)
        self.model = BackwardPassModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.model.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_hidden_states: bool | None = None,
        **kwargs,  # accepting **kwargs lets Trainer pass num_items_in_batch for exact gradient accumulation
    ) -> CausalLMOutputWithPast:
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
        )
        logits = self.lm_head(out.last_hidden_state)  # (B, T, V)
        loss = None
        if labels is not None:
            # The library's causal-LM loss: shift by one, ignore -100, divide by num_items_in_batch if given.
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
        return CausalLMOutputWithPast(
            loss=loss, logits=logits, past_key_values=out.past_key_values, hidden_states=out.hidden_states
        )


def register() -> None:
    """Teach the Auto classes about model_type="backward_pass". Safe to call more than once."""
    AutoConfig.register("backward_pass", BackwardPassConfig, exist_ok=True)
    AutoModel.register(BackwardPassConfig, BackwardPassModel, exist_ok=True)
    AutoModelForCausalLM.register(BackwardPassConfig, BackwardPassForCausalLM, exist_ok=True)
