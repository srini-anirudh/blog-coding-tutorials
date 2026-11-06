"""A toy of knowledge insulation: does training a new action expert damage what the backbone knew?

Scene: 4 objects, each with one of 20 colours and a 2-D position. Instruction: "go to the <colour> object".
Prefix tokens: one per object (colour embedding + position) and one instruction token.

Stage 1, "VLM pretraining": the backbone learns to point. A small head on the instruction token's last hidden state
predicts the named object's position, for all 20 colours.
Stage 2, "robot training": a randomly initialised action expert learns, by flow matching, an 8-step chunk that moves
the hand to the named object. Robot data only ever names colours 0 to 4.
Test: chunks for instructions naming colours 5 to 19, which only the pretraining saw.

Variants of stage 2:
  joint        backbone and expert trained together on the flow loss only
  joint+web    the same, plus the pointing loss on all colours as co-training data
  insulated    stop-gradient on the backbone keys and values the expert reads, plus pointing co-training
  frozen       backbone frozen
The real method also trains the backbone on discrete FAST action tokens; this toy leaves that part out.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .expert import ActionExpertPolicy

N_COLOURS, N_OBJ, H = 20, 4, 8
ROBOT_COLOURS = range(0, 5)


def make_batch(n: int, colours, gen: torch.Generator):
    """Random scenes whose named object has a colour from `colours`. The other objects have random distinct colours."""
    colours = torch.as_tensor(list(colours))
    target_colour = colours[torch.randint(len(colours), (n,), generator=gen)]
    obj = torch.stack([torch.randperm(N_COLOURS, generator=gen)[:N_OBJ] for _ in range(n)])
    slot = torch.randint(N_OBJ, (n,), generator=gen)
    # Put the target colour in its slot; if it was elsewhere, swap it out.
    clash = obj == target_colour[:, None]
    obj = torch.where(clash, obj[torch.arange(n), slot][:, None], obj)
    obj[torch.arange(n), slot] = target_colour
    pos = torch.rand(n, N_OBJ, 2, generator=gen) * 2 - 1
    target = pos[torch.arange(n), slot]
    hand = torch.rand(n, 2, generator=gen) * 2 - 1
    steps = torch.arange(1, H + 1, dtype=torch.float32)[None, :, None] / H
    chunk = hand[:, None] + (target - hand)[:, None] * steps  # straight line to the object
    return {"colours": obj, "pos": pos, "instr": target_colour, "target": target, "hand": hand, "chunk": chunk}


class ToyVLA(nn.Module):
    def __init__(self, d: int = 64, layers: int = 4):
        super().__init__()
        self.colour = nn.Embedding(N_COLOURS, d)
        self.word = nn.Embedding(N_COLOURS, d)
        self.pos_in = nn.Linear(2, d)
        self.token_type = nn.Parameter(torch.randn(2, d) * 0.02)
        self.policy = ActionExpertPolicy(act_dim=2, state_dim=2, horizon=H, layers=layers, d_vlm=d, d_exp=64,
                                         heads=4, head_dim=16, mlp_vlm=4 * d, mlp_exp=128)
        self.point = nn.Linear(d, 2)

    def prefix(self, b):
        objs = self.colour(b["colours"]) + self.pos_in(b["pos"]) + self.token_type[0]
        instr = self.word(b["instr"])[:, None] + self.token_type[1]
        return torch.cat([objs, instr], 1)

    def backbone_params(self):
        own = list(self.colour.parameters()) + list(self.word.parameters()) + list(self.pos_in.parameters())
        return own + [self.token_type] + [p for l in self.policy.layers for p in l.vlm.parameters()] + \
            list(self.point.parameters())

    def expert_params(self):
        ids = {id(p) for p in self.backbone_params()}
        return [p for p in self.parameters() if id(p) not in ids]

    def pointing_loss(self, b):
        _, x = self.policy.encode_prefix(self.prefix(b))
        return F.mse_loss(self.point(x[:, -1]), b["target"])

    @torch.no_grad()
    def pointing_error(self, b):
        _, x = self.policy.encode_prefix(self.prefix(b))
        return float((self.point(x[:, -1]) - b["target"]).norm(dim=-1).mean())

    @torch.no_grad()
    def action_error(self, b, gen=None):
        """Distance between the chunk's last position and the named object, after ten flow steps."""
        chunk = self.policy.sample(self.prefix(b), b["hand"], steps=10, generator=gen)
        return float((chunk[:, -1] - b["target"]).norm(dim=-1).mean())


def pretrain(model: ToyVLA, steps: int, seed: int = 0, batch: int = 256) -> list[float]:
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.backbone_params(), lr=1e-3)
    log = []
    for s in range(steps):
        loss = model.pointing_loss(make_batch(batch, range(N_COLOURS), gen))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if s % 200 == 0:
            log.append(float(loss.detach()))
    return log


def robot_train(model: ToyVLA, variant: str, steps: int, seed: int = 0, batch: int = 256) -> ToyVLA:
    model = copy.deepcopy(model)
    gen = torch.Generator().manual_seed(seed + 1)
    torch.manual_seed(seed + 1)
    train_backbone = variant != "frozen"
    params = model.expert_params() + (model.backbone_params() if train_backbone else [])
    opt = torch.optim.AdamW(params, lr=1e-3)
    for _ in range(steps):
        b = make_batch(batch, ROBOT_COLOURS, gen)
        prefix = model.prefix(b)
        if not train_backbone:
            prefix = prefix.detach()
        cache, _ = model.policy.encode_prefix(prefix)
        if not train_backbone:
            cache = [(k.detach(), v.detach()) for k, v in cache]
        loss = model.policy.flow_loss(None, b["hand"], b["chunk"], stop_grad=variant == "insulated", cache=cache)
        if variant in ("joint+web", "insulated"):
            loss = loss + model.pointing_loss(make_batch(batch, range(N_COLOURS), gen))
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


def evaluate(model: ToyVLA, seed: int = 123, n: int = 2000) -> dict:
    gen = torch.Generator().manual_seed(seed)
    seen = make_batch(n, ROBOT_COLOURS, gen)
    unseen = make_batch(n, range(5, N_COLOURS), gen)
    g = torch.Generator().manual_seed(seed)
    return {"action_err_seen": round(model.action_error(seen, g), 4),
            "action_err_unseen": round(model.action_error(unseen, g), 4),
            "pointing_err_all": round(model.pointing_error(make_batch(n, range(N_COLOURS), gen)), 4)}


def chance_error(seed: int = 7, n: int = 20000) -> float:
    """Mean distance from the hand's start to the target: the error of a chunk that does not move."""
    b = make_batch(n, range(N_COLOURS), torch.Generator().manual_seed(seed))
    return float((b["hand"] - b["target"]).norm(dim=-1).mean())


def wrong_object_error(seed: int = 7, n: int = 20000) -> float:
    """Mean distance between two random objects: the error of going confidently to the wrong one."""
    g = np.random.default_rng(seed)
    a, c = g.uniform(-1, 1, (n, 2)), g.uniform(-1, 1, (n, 2))
    return float(np.linalg.norm(a - c, axis=-1).mean())
