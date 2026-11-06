"""Small, readable pieces of a model that both reads and draws images.

Each module builds one mechanism from the post "Models That See and Draw" at toy scale:

- data.py         synthetic images: a two-mode "cat" image and a small shapes dataset
- flow.py         rectified flow: training loss and Euler sampler
- tokenizers.py   VQ (straight-through, commitment loss), low-dim normalised VQ, FSQ, token budgets
- transformer.py  one tiny transformer that takes any boolean attention mask
- priors.py       token priors: raster autoregression, MaskGIT decoding, classifier-free guidance
- hybrid.py       a Transfusion-style model: one trunk, a text loss and a flow loss
- readers.py      a caption-aligned "semantic" encoder versus a pixel tokenizer, probed both ways
"""
