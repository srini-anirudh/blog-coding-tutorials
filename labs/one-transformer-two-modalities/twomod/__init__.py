"""One Transformer, two modalities: a text Transformer and a ViT built from the same block.

bpe        a byte-level BPE tokenizer trained from scratch (the text front door)
front      token lookup, patchify, patch projection (the two ways vectors are made)
positions  sinusoidal tables, table interpolation, a neighbour score for learned tables
masks      causal, bidirectional, prefix-LM and padding masks
block      one attention + MLP block, shared by both models
models     TextTransformer, ViT and a small CNN baseline
train      a small image-classification loop for the ViT vs CNN experiment
hw         one line naming the hardware behind a number
"""

from .block import Attention, Block, BlockConfig, MLP
from .models import SmallCNN, TextTransformer, ViT

__all__ = ["Attention", "Block", "BlockConfig", "MLP", "SmallCNN", "TextTransformer", "ViT"]
