"""Preference optimisation of the tiny VLM: standard DPO, the no-image ablation, and mDPO's extra terms."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass

import torch

from .model import pad, seq_logprob
from .objectives import anchor_loss, conditional_loss, dpo_loss, margin_split
from .pairs import KINDS, Pairs
from .world import OBJECTS, crop, decode, null_image


@dataclass
class Variant:
    name: str
    use_image: bool = True      # False: every image replaced by the null image, in policy and reference
    cond: bool = False          # mDPO conditional term: same y_w, corrupted image
    anchor: bool = False        # mDPO anchor on the chosen reward
    corrupt: str = "crop_small"  # crop_small | crop_large | unrelated


VARIANTS = {
    "dpo": Variant("dpo"),
    "dpo_noimage": Variant("dpo_noimage", use_image=False),
    "dpo_anchor": Variant("dpo_anchor", anchor=True),
    "mdpo_noanchor": Variant("mdpo_noanchor", cond=True),
    "mdpo": Variant("mdpo", cond=True, anchor=True),
    "mdpo_crop_large": Variant("mdpo_crop_large", cond=True, anchor=True, corrupt="crop_large"),
    "mdpo_unrelated": Variant("mdpo_unrelated", cond=True, anchor=True, corrupt="unrelated"),
}


def corrupt_images(pairs: Pairs, mode: str, seed: int) -> torch.Tensor:
    """The rejected image I' for the conditional term, one per pair.

    crop_small  remove the evidence for one object that y_w mentions (one slot of eight)
    crop_large  remove the evidence for every object in the scene
    unrelated   the image of a different scene
    """
    rng, gen = random.Random(seed), torch.Generator().manual_seed(seed)
    out = []
    for i, (s, img) in enumerate(zip(pairs.scenes, pairs.imgs)):
        present = [OBJECTS.index(o) for o in s.facts]
        if mode == "crop_small":
            out.append(crop(img, [rng.choice(present)], gen))
        elif mode == "crop_large":
            out.append(crop(img, present, gen))
        elif mode == "unrelated":
            out.append(pairs.imgs[(i + 1 + rng.randrange(len(pairs) - 1)) % len(pairs)])
        else:
            raise ValueError(mode)
    return torch.stack(out)


@torch.no_grad()
def logps(model, imgs: torch.Tensor, seqs: list[list[int]], bs: int = 512) -> torch.Tensor:
    return torch.cat([seq_logprob(model, imgs[i:i + bs], pad(seqs[i:i + bs])) for i in range(0, len(seqs), bs)])


def nulls(n: int) -> torch.Tensor:
    return null_image().expand(n, -1, -1).clone()


def train_pref(ref, pairs: Pairs, v: Variant, steps: int = 300, bs: int = 64, lr: float = 1e-4,
               beta: float = 0.1, seed: int = 0, log=print):
    """Train a copy of `ref` on `pairs` with variant `v`. Returns (policy, log rows)."""
    torch.manual_seed(seed)
    policy = copy.deepcopy(ref).train()
    imgs = pairs.imgs if v.use_image else nulls(len(pairs))
    rw, rl = logps(ref, imgs, pairs.yw), logps(ref, imgs, pairs.yl)
    if v.cond:
        bad = corrupt_images(pairs, v.corrupt, seed + 1)
        rw_bad = logps(ref, bad, pairs.yw)
    opt = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=0.0)
    gen, rows = torch.Generator().manual_seed(seed), []
    for step in range(steps):
        j = torch.randint(len(pairs), (bs,), generator=gen)
        yw, yl = pad([pairs.yw[i] for i in j]), pad([pairs.yl[i] for i in j])
        pw, pl = seq_logprob(policy, imgs[j], yw), seq_logprob(policy, imgs[j], yl)
        loss, margin = dpo_loss(pw, pl, rw[j], rl[j], beta)
        total = loss
        if v.cond:
            total = total + conditional_loss(pw, seq_logprob(policy, bad[j], yw), rw[j], rw_bad[j], beta)
        if v.anchor:
            total = total + anchor_loss(pw, rw[j], beta)
        opt.zero_grad()
        total.backward()
        opt.step()
        if step % 50 == 0 or step == steps - 1:
            rows.append({"step": step, "dpo_loss": loss.item(), "total": total.item(), "margin": margin.mean().item()})
            log(f"[{v.name}] step {step:4d} dpo_loss={loss.item():.4f} total={total.item():.4f}")
    return policy.eval(), rows


@torch.no_grad()
def split_on(policy, ref, pairs: Pairs, beta: float = 0.1) -> dict[str, float]:
    """Where did the margin come from: the text part or the image part? Measured with the true images."""
    n = len(pairs)
    d = lambda m: {"w_img": logps(m, pairs.imgs, pairs.yw), "w_null": logps(m, nulls(n), pairs.yw),  # noqa: E731
                   "l_img": logps(m, pairs.imgs, pairs.yl), "l_null": logps(m, nulls(n), pairs.yl)}
    return margin_split(d(policy), d(ref), beta)


@torch.no_grad()
def fact_accuracy(policy, ref, pairs: Pairs) -> dict[str, float]:
    """On pairs that only the image separates: does the policy prefer y_w, and did training move it there?"""
    pw, pl = logps(policy, pairs.imgs, pairs.yw), logps(policy, pairs.imgs, pairs.yl)
    rw, rl = logps(ref, pairs.imgs, pairs.yw), logps(ref, pairs.imgs, pairs.yl)
    kinds = torch.tensor([KINDS.index(k) for k in pairs.kinds])
    out = {"policy_pref_acc": (pw > pl).float().mean().item(),
           "implicit_reward_acc": ((pw - rw) > (pl - rl)).float().mean().item()}
    for i, k in enumerate(KINDS):
        m = kinds == i
        out[f"policy_pref_acc_{k}"] = (pw[m] > pl[m]).float().mean().item()
    gw = pw - logps(policy, nulls(len(pairs)), pairs.yw)
    out["image_gain_yw"] = gw.mean().item()
    return out


def describe_ids(ids: list[int]) -> str:
    claims, _ = decode(ids)
    return " ; ".join(f"{c.count} {c.colour} {c.obj}" for c in claims)
