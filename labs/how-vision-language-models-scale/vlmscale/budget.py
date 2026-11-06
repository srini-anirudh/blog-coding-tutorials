"""The compute budget of one VLM training sample.

The post's accounting, in code:

    P = (H / p) * (W / p)                patches the encoder sees
    V = P / m**2                         visual tokens the LLM sees after an m x m merge
    C = k_V * N_V * P + k_L * N_L * (V + T)

k is FLOPs per parameter per token: 6 for a trained tower, 2 for a frozen tower that nothing below
needs gradients through, 4 for a frozen tower whose input still needs a gradient (a frozen LLM above
a trained connector). Attention-score FLOPs are not in this formula; `attention_flops` adds them.
"""

from __future__ import annotations

from dataclasses import dataclass

# FLOPs per parameter per token. Forward is 2. Backward is 2 for the input gradient and 2 for the
# weight gradient. A frozen tower skips the weight gradient; a tower nothing below it trains skips both.
K_TRAINED = 6
K_FROZEN_PASS_GRAD = 4
K_FROZEN = 2


def num_patches(height: int, width: int | None = None, patch: int = 14) -> int:
    width = height if width is None else width
    return (height // patch) * (width // patch)


def num_visual_tokens(patches: int, merge: int = 1) -> int:
    if patches % (merge * merge):
        raise ValueError(f"{patches} patches do not divide into {merge}x{merge} blocks")
    return patches // (merge * merge)


@dataclass
class SampleFlops:
    encoder: float
    llm: float
    connector: float
    patches: int
    visual_tokens: int

    @property
    def total(self) -> float:
        return self.encoder + self.llm + self.connector

    @property
    def encoder_share(self) -> float:
        return self.encoder / self.total


def sample_flops(
    n_v: float,
    n_l: float,
    side: int,
    patch: int = 14,
    merge: int = 1,
    text_tokens: int = 0,
    tiles: int = 1,
    k_v: int = K_TRAINED,
    k_l: int = K_TRAINED,
    n_c: float = 0.0,
    k_c: int = K_TRAINED,
) -> SampleFlops:
    """Weight-matmul FLOPs for one sample: `tiles` square crops of `side` pixels plus `text_tokens`."""
    p = num_patches(side, side, patch) * tiles
    v = num_visual_tokens(p, merge)
    return SampleFlops(
        encoder=k_v * n_v * p,
        llm=k_l * n_l * (v + text_tokens),
        connector=k_c * n_c * v,  # the connector runs once per merged token
        patches=p,
        visual_tokens=v,
    )


def encoder_share(n_v: float, n_l: float, merge: int = 1, text_per_visual: float = 0.0) -> float:
    """The post's closed form: N_V m^2 / (N_V m^2 + N_L (1 + T/V)), both towers trained."""
    a = n_v * merge * merge
    return a / (a + n_l * (1.0 + text_per_visual))


def llm_equivalent_of_encoder(n_v: float, merge: int) -> float:
    """Per image, an encoder of N_V with an m x m merge costs what an LLM of N_V * m^2 would."""
    return n_v * merge * merge


def attention_flops(seq: int, d_attn: int, layers: int, k: int = K_TRAINED, window: int | None = None,
                    full_layers: int | None = None) -> float:
    """Attention-score FLOPs (QK^T and AV) for one sequence, all layers.

    Forward per layer: 2*s*s*d for QK^T plus 2*s*s*d for AV = 4 s^2 d. Training multiplies by k/2
    (3x for a trained tower). With `window`, every layer except `full_layers` attends only inside
    windows of `window` tokens, so its cost is 4 * s * window * d.
    """
    per_full = 4.0 * seq * seq * d_attn
    if window is None:
        return k / 2 * per_full * layers
    full = layers if full_layers is None else full_layers
    per_window = 4.0 * seq * min(window, seq) * d_attn
    return k / 2 * (per_full * full + per_window * (layers - full))


def weight_flops(n_params: float, tokens: int, k: int = K_TRAINED) -> float:
    return k * n_params * tokens


# Published configurations used by the scripts. Parameter counts in billions, as quoted in the post.
PRESETS = {
    "fig1-7B": dict(n_v=0.4, n_l=7.0, side=448, patch=14, merge=2, text_tokens=200),
    "fig1-0.5B": dict(n_v=0.4, n_l=0.5, side=448, patch=14, merge=2, text_tokens=200),
    "Qwen2-VL-2B": dict(n_v=0.675, n_l=1.5, merge=2),
    "Qwen2-VL-7B": dict(n_v=0.675, n_l=7.6, merge=2),
    "Qwen2-VL-72B": dict(n_v=0.675, n_l=72.0, merge=2),
    "SmolVLM-256M (r=4)": dict(n_v=0.093, n_l=0.135, merge=4),
    "SmolVLM-256M (r=2)": dict(n_v=0.093, n_l=0.135, merge=2),
    "LLaVA-1.5-13B": dict(n_v=0.3, n_l=13.0, merge=1),
    "InternVL2.5-78B": dict(n_v=5.5, n_l=72.0, merge=2),
}
