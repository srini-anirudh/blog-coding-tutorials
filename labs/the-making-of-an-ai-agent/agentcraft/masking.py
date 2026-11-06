"""Mask the world's tokens: what goes wrong when SFT on a trajectory also scores the observations.

One synthetic episode is a flat token sequence:

    <bos> task K3 <act> search K3 <eos> <obs> w w w K7 V2 w w K3 V9 w ... <act> answer K3 V9 <eos>
                       '-- model --'          '----------- world (about 94%) ----------'  '--- model ---'

The model writes two actions. The world writes one long observation: words from a fixed Markov chain (the tool's
"grammar") with a few key-value facts mixed in. The right answer restates the key and copies the value that follows it in the
observation (the copy an induction head does). Facts are
written over aligned pairs of passage positions.

Two objectives, same model, same data, same steps:
    masked   : cross entropy on the model-written tokens only (the policy's likelihood)
    unmasked : cross entropy on every token (also fits P(o | s, a), a model of the world)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

SPECIALS = ["<pad>", "<bos>", "<act>", "<eos>", "<obs>", "task", "search", "answer"]
N_KEYS, N_VALUES, N_WORDS = 16, 16, 48
KEYS = [f"K{i}" for i in range(N_KEYS)]
VALUES = [f"V{i}" for i in range(N_VALUES)]
WORDS = [f"w{i}" for i in range(N_WORDS)]
VOCAB = SPECIALS + KEYS + VALUES + WORDS
ID = {w: i for i, w in enumerate(VOCAB)}
PAD = ID["<pad>"]


@dataclass
class EpisodeConfig:
    obs_len: int = 110     # tokens of world output per episode
    n_facts: int = 4       # key-value facts hidden in the observation, one of them is the answer
    branching: int = 4     # each passage word has this many possible successors


def world_grammar(cfg: EpisodeConfig, seed: int = 0) -> torch.Tensor:
    """(N_WORDS, branching) successor table: the fixed, learnable structure of the tool's output."""
    g = torch.Generator().manual_seed(seed)
    return torch.stack([torch.randperm(N_WORDS, generator=g)[: cfg.branching] for _ in range(N_WORDS)])


def make_batch(n: int, cfg: EpisodeConfig, grammar: torch.Tensor, g: torch.Generator):
    """Returns tokens (n, T), action_mask (n, T) marking model-written targets, and answer positions (n,).

    Layout (fixed length T = obs_len + 13):
        <bos> task K <act> | search K <eos> | <obs> w ... w | <act> | answer K V <eos>
    """
    off = len(SPECIALS) + N_KEYS + N_VALUES                       # id of the first passage word
    # the observation: a Markov-chain passage over the words, one step per position, vectorised over the batch
    words = torch.empty(n, cfg.obs_len, dtype=torch.long)
    words[:, 0] = torch.randint(N_WORDS, (n,), generator=g)
    picks = torch.randint(cfg.branching, (n, cfg.obs_len), generator=g)
    for t in range(1, cfg.obs_len):
        words[:, t] = grammar[words[:, t - 1], picks[:, t]]
    obs = words + off
    # key-value facts written over random aligned pairs of positions; fact 0 is the one the task asks about
    keys = torch.rand(n, N_KEYS, generator=g).argsort(1)[:, : cfg.n_facts]
    vals = torch.randint(N_VALUES, (n, cfg.n_facts), generator=g)
    slots = torch.rand(n, cfg.obs_len // 2, generator=g).argsort(1)[:, : cfg.n_facts] * 2
    rows = torch.arange(n)[:, None]
    obs[rows, slots] = keys + len(SPECIALS)
    obs[rows, slots + 1] = vals + len(SPECIALS) + N_KEYS
    key, val = keys[:, 0] + len(SPECIALS), vals[:, 0] + len(SPECIALS) + N_KEYS

    def col(x):
        return torch.full((n,), ID[x]) if isinstance(x, str) else x

    head = torch.stack([col(x) for x in ["<bos>", "task", key, "<act>", "search", key, "<eos>", "<obs>"]], 1)
    tail = torch.stack([col(x) for x in ["<act>", "answer", key, val, "<eos>"]], 1)
    tokens = torch.cat([head, obs, tail], 1)
    action_mask = torch.zeros(tokens.shape)
    action_mask[:, 4:7] = 1                                        # search K <eos>
    action_mask[:, -4:] = 1                                        # answer K V <eos>
    return tokens, action_mask, torch.full((n,), tokens.shape[1] - 2)


class TinyLM(nn.Module):
    """A 2-layer causal transformer. Small on purpose: capacity is the thing the two objectives compete for."""

    def __init__(self, d: int = 64, layers: int = 2, heads: int = 4, max_len: int = 160):
        super().__init__()
        self.tok = nn.Embedding(len(VOCAB), d)
        self.pos = nn.Embedding(max_len, d)
        block = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, len(VOCAB))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[1]
        causal = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        h = self.tok(x) + self.pos(torch.arange(T, device=x.device))
        return self.head(self.norm(self.blocks(h, mask=causal, is_causal=True)))


def token_nll(model: TinyLM, tokens: torch.Tensor) -> torch.Tensor:
    """(n, T-1) negative log-likelihood of each next token."""
    logits = model(tokens[:, :-1])
    return F.cross_entropy(logits.transpose(1, 2), tokens[:, 1:], reduction="none")


def losses(model: TinyLM, tokens: torch.Tensor, action_mask: torch.Tensor):
    """Mean NLL over model-written tokens, and over world-written tokens (pads and the prompt excluded)."""
    nll = token_nll(model, tokens)
    act = action_mask[:, 1:]
    world = (tokens[:, 1:] != PAD).float() * (1 - act)
    return (nll * act).sum() / act.sum(), (nll * world).sum() / world.sum(), act.sum(), world.sum()


def objective(model, tokens, action_mask, masked: bool) -> torch.Tensor:
    l_act, l_world, n_act, n_world = losses(model, tokens, action_mask)
    if masked:
        return l_act
    return (l_act * n_act + l_world * n_world) / (n_act + n_world)   # plain next-token loss on everything


@torch.no_grad()
def evaluate(model: TinyLM, tokens, action_mask, answer_pos) -> dict:
    model.eval()
    logits = model(tokens[:, :-1])
    rows = torch.arange(len(tokens), device=tokens.device)
    pred = logits[rows, answer_pos - 1].argmax(-1)          # the value token after "answer K"
    l_act, l_world, n_act, n_world = losses(model, tokens, action_mask)
    model.train()
    return {"answer_acc": (pred == tokens[rows, answer_pos]).float().mean().item(),
            "action_nll": l_act.item(), "world_nll": l_world.item(),
            "action_share": (n_act / (n_act + n_world)).item()}


def gradient_split(model: TinyLM, tokens, action_mask) -> dict:
    """How much of the unmasked gradient is the world's. Returns the cosine between the unmasked gradient and the
    policy (masked) gradient, and the norm of the world part relative to the policy part."""
    params = [p for p in model.parameters() if p.requires_grad]
    l_act, l_world, n_act, n_world = losses(model, tokens, action_mask)
    n = n_act + n_world
    g_act = torch.autograd.grad(l_act * n_act / n, params, retain_graph=True)
    g_world = torch.autograd.grad(l_world * n_world / n, params)
    ga = torch.cat([g.flatten() for g in g_act])
    gw = torch.cat([g.flatten() for g in g_world])
    return {"cos_unmasked_vs_policy": F.cosine_similarity(ga + gw, ga, dim=0).item(),
            "world_over_policy_norm": (gw.norm() / ga.norm()).item()}


def train(masked: bool, steps: int, seed: int, cfg: EpisodeConfig | None = None, batch: int = 32,
          lr: float = 3e-3, eval_every: int = 250, n_eval: int = 512, device: str = "cpu", log=print) -> dict:
    cfg = cfg or EpisodeConfig()
    torch.manual_seed(seed)
    grammar = world_grammar(cfg)
    g_train = torch.Generator().manual_seed(1000 + seed)
    eval_set = [t.to(device) for t in make_batch(n_eval, cfg, grammar, torch.Generator().manual_seed(999))]
    model = TinyLM().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.1)
    history = []
    for step in range(1, steps + 1):
        tokens, action_mask, _ = (t.to(device) for t in make_batch(batch, cfg, grammar, g_train))
        loss = objective(model, tokens, action_mask, masked)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % eval_every == 0 or step == steps:
            row = {"step": step, **evaluate(model, *eval_set)}
            if not masked:
                row.update(gradient_split(model, tokens, action_mask))
            history.append(row)
            log(f"{'masked' if masked else 'unmasked'} seed {seed} " + " ".join(
                f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()))
    return {"masked": masked, "seed": seed, "history": history, "final": history[-1]}
