"""vitfamily: the mechanisms of "An Image Is a Sentence", built small.

patch.py     patchify as a strided conv, patch merging, tubelets, position-table resizing
attention.py global MHSA, PVT spatial-reduction attention, MViT pooling attention, ConViT GPSA
blocks.py    pre-norm ViT block, Swin W-MSA / SW-MSA block, divided space-time block
models.py    TinyViT (CLS, GAP, distillation token, GPSA layers) and TinyCNN
flops.py     the post's FLOP and memory formulas, and a counter that measures real modules
reach.py     who-can-reach-whom over depth: global, windows, shifted windows, axial, conv
stats.py     mean attention distance
data.py      CIFAR-10 and tiny-imagenet from Hugging Face parquet files

Swin's window partition, shift mask and window attention come from the C3 tutorial (attnzoo).
"""
