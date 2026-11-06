"""Loss weights correct for counting: Emu3's 0.5 and Transfusion's 5.

Part 1 is arithmetic. A 512 x 512 image with a 30-token caption:
  - Emu3 style: every image token is its own cross-entropy term, like a text token.
  - Transfusion style: the diffusion loss is one mean over the image's patches.
It prints the image's share of the loss terms, before and after each paper's weight.

Part 2 measures the same thing in this lab's hybrid model: the gradient norm on the shared trunk from the
text loss and from the weighted flow loss, at initialisation, for several lambda values, and for a
"sum over patch values" normalisation of the flow loss.
"""

from __future__ import annotations

import torch

import _shared
from seedraw import hybrid
from seedraw.common import Report, seed_all


def share(image_weight: float, text_weight: float) -> float:
    return image_weight / (image_weight + text_weight)


def trunk_grad_norm(model, loss) -> float:
    model.zero_grad()
    loss.backward(retain_graph=True)
    return float(torch.sqrt(sum(p.grad.pow(2).sum() for p in model.trunk.parameters() if p.grad is not None)))


def main() -> None:
    seed_all(0)
    rep = Report("07_loss_balance")
    text, img_tokens = 30, 4096
    rep("Part 1: a 512 x 512 image (Emu3: 4,096 tokens) and a 30-token caption.")
    rows = {
        "per-token CE, no weight": share(img_tokens, text),
        "per-token CE, image x 0.5 (Emu3)": share(0.5 * img_tokens, text),
        "one diffusion mean per image, no weight": share(1, text),
        "one diffusion mean per image, x 5 (Transfusion)": share(5, text),
    }
    for k, v in rows.items():
        rep(f"  {k:50s} image share of loss terms {v:6.1%}")
    rep("  Per-token counting puts about 99% of the loss terms on the image. A weight of 0.5 trims that share only a little "
        "(by the term count alone). One mean per image starts near 3%, and a weight of 5 raises it.")
    rep("  The two weights correct opposite counting rules. Term counts are not gradient shares, so Part 2 measures gradients.")
    rep()

    train, _ = _shared.splits()
    x = train["images"][:256]
    rep("Part 2: trunk gradient norms in this lab's hybrid at initialisation, batch 256 (16 patches, 2-3 text targets).")
    rep(f"  {'flow loss normalisation':28s} {'lambda':>6s} {'|g| text':>9s} {'|g| flow':>9s} {'flow / text':>11s}")
    out = []
    for norm in ("mean over patches", "sum over patch values"):
        for lam in (1.0, 5.0):
            seed_all(0)
            model = hybrid.HybridModel()
            torch.manual_seed(1)
            losses = hybrid.hybrid_losses(model, x, train["shape"][:256], train["side"][:256])
            flow = losses["flow"] * (hybrid.N_PATCH * hybrid.PATCH_DIM if norm.startswith("sum") else 1)
            g_text = trunk_grad_norm(model, losses["lm"])
            g_flow = trunk_grad_norm(model, lam * flow)
            rep(f"  {norm:28s} {lam:6.1f} {g_text:9.4f} {g_flow:9.4f} {g_flow / g_text:11.2f}")
            out.append({"normalisation": norm, "lambda": lam, "grad_text": g_text, "grad_flow": g_flow})
    rep("  Check how a loss is normalised before copying its weight from a paper.")
    rep.data = {"part1_image_share": rows, "part2": out}
    rep.save()


if __name__ == "__main__":
    main()
