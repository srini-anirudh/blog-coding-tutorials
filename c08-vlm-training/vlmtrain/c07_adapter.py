"""Train C7's model. C7 builds `tinyvlm`; this file is the only place C8 touches its internals.

The trainer needs three things from a model: submodules `.vision`, `.projector`, `.lm`, and
forward/generate with the C7 signature. C7 provides all of them. Only the tokenizer needs more:
C7 adds <image>, and grounding also needs the <loc_k> tokens.
"""

from __future__ import annotations

import sys
from pathlib import Path

C07_DIR = Path(__file__).resolve().parents[2] / "c07-vlm"


def c07_available() -> bool:
    return (C07_DIR / "tinyvlm" / "__init__.py").exists()


def _import_c07() -> None:
    if not c07_available():
        raise FileNotFoundError(f"{C07_DIR} not found. Use model=own, which needs nothing from C7.")
    if str(C07_DIR) not in sys.path:
        sys.path.insert(0, str(C07_DIR))


def _finish(model, c07_processor, strategy: str, max_tiles: int, bins: int):
    """Grow the vocabulary for the location tokens and wrap the model with our unified-sample processor."""
    from .processor import TinyVLMProcessor
    from .tokenizer import add_vlm_tokens

    tok = add_vlm_tokens(c07_processor.tokenizer, bins)
    # C7 pads the embedding matrix to a multiple of 64, so the real model already has spare rows.
    if model.lm.get_input_embeddings().num_embeddings < len(tok):
        model.lm.resize_token_embeddings(len(tok), mean_resizing=True)
    proc = TinyVLMProcessor(tok, model.config.image_seq_len, model.config.vision_config.image_size,
                            strategy, max_tiles)
    return tok, model.train(), proc


def build_c07(strategy: str = "tile", max_tiles: int = 4, bins: int = 32):
    """Pretrained SigLIP + fresh projector + pretrained SmolLM2, through C7's own builder."""
    _import_c07()
    from tinyvlm import build_tinyvlm

    model, c07_processor = build_tinyvlm(pixel_shuffle_factor=2)
    return _finish(model, c07_processor, strategy, max_tiles, bins)


def build_c07_tiny(strategy: str = "tile", max_tiles: int = 4, bins: int = 32, seed: int = 0):
    """C7's random tiny model. Same classes, no downloads, used by the tests."""
    _import_c07()
    from tinyvlm.testing import tiny_vlm

    model, c07_processor = tiny_vlm(pixel_shuffle_factor=2, seed=seed)
    return _finish(model, c07_processor, strategy, max_tiles, bins)
