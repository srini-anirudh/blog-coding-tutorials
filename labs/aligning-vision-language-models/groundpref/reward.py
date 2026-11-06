"""A Bradley-Terry reward model r(I, x, y) and the image-ablation audit from the post.

The reward model is the same tiny VLM with a scalar head read at the last response token. It can see the image.
Whether it uses the image depends only on the pairs it is trained on.
"""

from __future__ import annotations

import random

import torch
import torch.nn.functional as F_

from .model import TinyVLM, pad
from .pairs import Pairs
from .world import PAD, decode, edit_colour, render


def reward(rm: TinyVLM, imgs: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """(B,) scalar reward at the last real token."""
    h = rm(imgs, ids).squeeze(-1)                 # (B, T)
    last = (ids != PAD).sum(1) - 1
    return h.gather(1, last[:, None]).squeeze(1)


@torch.no_grad()
def rewards(rm, imgs, seqs, bs: int = 512) -> torch.Tensor:
    return torch.cat([reward(rm, imgs[i:i + bs], pad(seqs[i:i + bs])) for i in range(0, len(seqs), bs)])


def train_rm(pairs: Pairs, steps: int = 1500, bs: int = 128, lr: float = 1e-3, seed: int = 0) -> TinyVLM:
    """L = -log sigma(r(I, y_w) - r(I, y_l))."""
    torch.manual_seed(seed)
    rm = TinyVLM(reward_head=True)
    opt = torch.optim.AdamW(rm.parameters(), lr=lr, weight_decay=0.01)
    gen = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        j = torch.randint(len(pairs), (bs,), generator=gen)
        rw = reward(rm, pairs.imgs[j], pad([pairs.yw[i] for i in j]))
        rl = reward(rm, pairs.imgs[j], pad([pairs.yl[i] for i in j]))
        loss = -F_.logsigmoid(rw - rl).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return rm.eval()


@torch.no_grad()
def audit(rm, pairs: Pairs, seed: int = 0, noise: float = 0.8) -> dict[str, float]:
    """Score with the true image, no image and a swapped image.

    sensitivity  E|r(I, y) - r(null, y)| divided by the spread (sd) of r(I, y): 0 means the image is ignored.
    swap_flip    on colour pairs, edit the image so the REJECTED colour becomes true. A grounded reward
                 should now prefer y_l. Fraction of pairs where the preference flips.
    """
    from .prefopt import nulls

    n = len(pairs)
    r_img = rewards(rm, torch.cat([pairs.imgs, pairs.imgs]), pairs.yw + pairs.yl)
    r_null = rewards(rm, torch.cat([nulls(n), nulls(n)]), pairs.yw + pairs.yl)
    sens = ((r_img - r_null).abs().mean() / r_img.std()).item()
    acc = (r_img[:n] > r_img[n:]).float().mean().item()

    gen, idx, edited = torch.Generator().manual_seed(seed), [], []
    for i, (s, k) in enumerate(zip(pairs.scenes, pairs.kinds)):
        if k != "colour":
            continue
        cw, _ = decode(pairs.yw[i])
        cl, _ = decode(pairs.yl[i])
        diff = [(a, b) for a, b in zip(cw, cl) if a != b]
        if len(diff) != 1:
            continue
        idx.append(i)
        edited.append(render(edit_colour(s, diff[0][1].obj, diff[0][1].colour), gen, noise))
    sw = rewards(rm, torch.stack(edited), [pairs.yw[i] for i in idx])
    sl = rewards(rm, torch.stack(edited), [pairs.yl[i] for i in idx])
    ow, ol = r_img[idx], r_img[[n + i for i in idx]]
    before = ow > ol
    flip = ((sl > sw) & before).float().sum().item() / max(1, before.sum().item())
    out = {"pair_acc": acc, "sensitivity": sens, "swap_flip": flip, "swap_pairs": len(idx)}
    for k in sorted(set(pairs.kinds)):
        m = torch.tensor([x == k for x in pairs.kinds])
        out[f"pair_acc_{k}"] = (r_img[:n][m] > r_img[n:][m]).float().mean().item()
    return out


def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
