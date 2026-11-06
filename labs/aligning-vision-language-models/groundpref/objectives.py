"""Preference objectives written out term by term.

Every input below is a (B,) tensor of sequence log-probabilities:
    pw, pl   log pi_theta(y_w | x, I), log pi_theta(y_l | x, I)
    rw, rl   the same under the frozen reference pi_ref
"""

from __future__ import annotations

import difflib

import torch
import torch.nn.functional as F_


def dpo_loss(pw, pl, rw, rl, beta: float = 0.1) -> tuple[torch.Tensor, torch.Tensor]:
    """-log sigma(beta [(pw - rw) - (pl - rl)]). Returns (mean loss, per-pair margin)."""
    margin = beta * ((pw - rw) - (pl - rl))
    return -F_.logsigmoid(margin).mean(), margin.detach()


def conditional_loss(pw_img, pw_bad, rw_img, rw_bad, beta: float = 0.1) -> torch.Tensor:
    """mDPO's image-side term: the SAME response y_w, true image I against corrupted image I'.

    The text-only parts log pi(y_w | x) appear on both sides and cancel, so only the image can lower this loss.
    """
    margin = beta * ((pw_img - rw_img) - (pw_bad - rw_bad))
    return -F_.logsigmoid(margin).mean()


def anchor_loss(pw, rw, beta: float = 0.1, delta: float = 0.0) -> torch.Tensor:
    """mDPO's anchor: keep the chosen response's implicit reward above delta, so DPO cannot push both down."""
    return -F_.logsigmoid(beta * (pw - rw) - delta).mean()


def margin_split(pol: dict, ref: dict, beta: float = 0.1) -> dict[str, float]:
    """Split the DPO margin into a text part and an image part.

    pol and ref hold (B,) log-probs under keys 'w_img', 'w_null', 'l_img', 'l_null'
    (chosen or rejected, with the true image or the null image). With g = log p(y|I) - log p(y|null):

        margin = beta (D_text + D_image)
        D_text  = [pol_w_null - ref_w_null] - [pol_l_null - ref_l_null]
        D_image = [g_pol(w) - g_ref(w)] - [g_pol(l) - g_ref(l)]
    """
    g = lambda d, s: d[f"{s}_img"] - d[f"{s}_null"]  # noqa: E731
    d_text = (pol["w_null"] - ref["w_null"]) - (pol["l_null"] - ref["l_null"])
    d_image = (g(pol, "w") - g(ref, "w")) - (g(pol, "l") - g(ref, "l"))
    total = (pol["w_img"] - ref["w_img"]) - (pol["l_img"] - ref["l_img"])
    return {"beta_d_text": beta * d_text.mean().item(), "beta_d_image": beta * d_image.mean().item(),
            "beta_margin": beta * total.mean().item()}


def changed_mask(a: list[int], b: list[int]) -> tuple[list[bool], list[bool]]:
    """Which tokens of a and of b sit inside an edit, according to a longest-matching-block diff."""
    ma, mb = [True] * len(a), [True] * len(b)
    for blk in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_matching_blocks():
        for i in range(blk.size):
            ma[blk.a + i] = False
            mb[blk.b + i] = False
    return ma, mb


def dense_weights(ids: torch.Tensor, changed: list[list[bool]], gamma: float) -> torch.Tensor:
    """(B, T-1) per-target weights: gamma on edited tokens, 1 elsewhere (RLHF-V Dense DPO)."""
    w = torch.ones(ids.shape[0], ids.shape[1] - 1)
    for i, ch in enumerate(changed):
        for t, c in enumerate(ch[1:]):  # target position t+1 lives at column t
            if c:
                w[i, t] = gamma
    return w
