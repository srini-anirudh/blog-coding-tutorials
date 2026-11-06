"""Section 12: the VLM modality interface as a normalization problem, at toy scale.

Part A (Fig 11). Representational inertia. In a Pre-LN block the update F(Norm(h)) does not depend on ||h||.
So the same update turns a short vector a lot and a long vector very little:
    tan(theta) = C sin(phi) / (||h|| + C cos(phi)),   C = ||update||, phi = angle between h and the update.
We check the formula, then push one direction at text-like and vision-like norms through one Pre-LN MLP block.
Part B. A LayerNorm after the projector, with its gain set to T / sqrt(D) so visual tokens land at text norm T.
The gradient that reaches the encoder is multiplied by that small gain. A backward hook that scales the gradient
by 1 / mean(gain) gives it back, without changing the forward pass.
The norms 0.8, 29 and 72 are the post's measured values for text embeddings, CLIP ViT-L/14 and SigLIP-SO400M.
"""

import math

import _setup  # noqa: F401
import torch
import torch.nn.functional as F
from torch import nn

from normgeo.blocks import MLP
from normgeo.geometry import GradScale, rotation_angle
from normgeo.ops import RMSNorm
from normgeo.report import Report

torch.manual_seed(0)
D = 576  # SmolLM2-135M's hidden size; any width shows the same thing
NORMS = [0.8, 5.0, 29.0, 72.0]


def angle_deg(a: torch.Tensor, b: torch.Tensor) -> float:
    return math.degrees(math.acos(F.cosine_similarity(a, b, dim=-1).clamp(-1, 1).mean().item()))


with Report("09_interface") as say:
    say("Part A. One update of norm C = 1, at angle phi to h. Measured angle vs the formula, degrees.")
    say(f"{'||h||':>7}{'phi':>6}{'measured':>10}{'formula':>9}")
    for n in NORMS:
        for phi in [90.0, 45.0]:
            h = F.normalize(torch.randn(D), dim=0)
            ortho = F.normalize(torch.randn(D) - (torch.randn(D) @ h) * h, dim=0)
            ortho = F.normalize(ortho - (ortho @ h) * h, dim=0)
            r = math.radians(phi)
            u = math.cos(r) * h + math.sin(r) * ortho  # unit update at angle phi to h
            meas = angle_deg(n * h, n * h + u)
            pred = math.degrees(rotation_angle(n, 1.0, r))
            say(f"{n:>7}{phi:>6.0f}{meas:>10.3f}{pred:>9.3f}")

    say()
    say("  The same direction at four norms through one random Pre-LN MLP block, x + MLP(RMSNorm(x)):")
    block_norm, mlp = RMSNorm(D), MLP(D)
    dirs = F.normalize(torch.randn(256, D), dim=-1)
    rows = []
    with torch.no_grad():
        for n in NORMS:
            x = dirs * n
            upd = mlp(block_norm(x))
            rows.append({"norm": n, "update_norm": upd.norm(dim=-1).mean().item(), "angle_deg": angle_deg(x, x + upd)})
            say(f"  ||h|| {n:>5}:  ||update|| {rows[-1]['update_norm']:.3f}   rotation {rows[-1]['angle_deg']:.2f} deg")
    say("  The update is identical in size at every norm. The rotation it causes falls roughly as 1 / ||h||.")

    say()
    T, D_ENC = 0.8, 64
    g0 = T / math.sqrt(D)
    say(f"Part B. encoder Linear({D_ENC}, {D}) -> projector Linear({D}, {D}) -> interface. Encoder outputs start at norm ~72.")
    say(f"  A fixed upstream gradient G (unit norm per token) stands in for the LLM. Gain init T/sqrt(D) = {g0:.4f}.")
    say(f"{'interface':<34}{'visual norm':>12}{'ratio to text':>15}{'||grad encoder||':>18}")
    x_in = torch.randn(128, D_ENC)
    G = F.normalize(torch.randn(128, D), dim=-1)
    out = []
    for name in ["bare projector", "LayerNorm, gain 1", "LayerNorm, gain T/sqrt(D)", "LayerNorm, gain T/sqrt(D) + hook"]:
        torch.manual_seed(1)
        enc, proj = nn.Linear(D_ENC, D), nn.Linear(D, D)
        with torch.no_grad():
            enc.weight.mul_(72 / enc(x_in).norm(dim=-1).mean())  # put visual features at norm ~72
        ln = nn.LayerNorm(D)
        if "T/sqrt" in name:
            with torch.no_grad():
                ln.weight.fill_(g0)
        h = proj(enc(x_in))
        if name != "bare projector":
            if "hook" in name:
                h = GradScale.apply(h, 1.0 / ln.weight.mean().item())
            h = ln(h)
        (h * G).sum().backward()
        vnorm = h.norm(dim=-1).mean().item()
        gn = enc.weight.grad.norm().item()
        out.append({"interface": name, "visual_norm": vnorm, "encoder_grad": gn})
        say(f"{name:<34}{vnorm:>12.2f}{vnorm / T:>15.1f}{gn:>18.3f}")
    say("  The small gain fixes the norm ratio but cuts the encoder's gradient by the same factor. The hook restores it.")
    say.json({"part_a_block": rows, "part_b": out})
