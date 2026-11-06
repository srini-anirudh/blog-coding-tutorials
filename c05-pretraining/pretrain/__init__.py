"""pretrain: a small, complete LLM pretraining system.

corpus -> tokenizer -> EOS-separated token shards -> packed windows -> resumable loader
-> AdamW + warmup/cosine -> accumulation, bf16, clipping, activation checkpointing
-> full checkpoints and exact resume -> logging and MFU -> DDP / FSDP / ZeRO -> failure guards.
"""

from pretrain.config import ModelArgs, TrainConfig

__all__ = ["ModelArgs", "TrainConfig"]
