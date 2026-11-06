"""A tiny chart-reading VLM, trained from scratch, for the post's experiments on one visual question type.

The image is a two-bar chart. The question asks for the SUM or the absolute
DIFFerence of the two bars. The policy answers in three tokens:

    [ visual tokens (36) | question | BOS ]  ->  read1  read2  answer
        E: CNN encoder      P: connector (linear)      LM: 2-layer causal transformer

The two reading tokens are perception, the answer token is reasoning. That split
lets us run the post's controls exactly:
    real image          A_img     answer accuracy
    blank image         A_blank   answer accuracy from priors alone
    oracle perception   A_oracle  readings forced to the true values, answer generated
    perception probe    P         both readings correct

The C7 tiny VLM is a full image-to-text model. This one is much smaller on purpose:
a GRPO step over 256 rollouts takes milliseconds on a CPU, so every
experiment runs with three seeds.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from .grpo import group_advantages, zero_advantage

N_VALUES = 10                    # bar values 0..9
N_NUM = 19                       # number tokens 0..18 (a sum can reach 18)
SUM, DIFF, BOS = 19, 20, 21
VOCAB = 22
SIZE = 24                        # image side in pixels
N_VIS = 36                       # 6 x 6 visual tokens
PARTS = ("encoder", "connector", "llm")


# ---------- data ----------

def answer_of(q: torch.Tensor, v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
    return torch.where(q == SUM, v1 + v2, (v1 - v2).abs())


def render(v1: torch.Tensor, v2: torch.Tensor, g: torch.Generator, noise: float = 0.35) -> torch.Tensor:
    """(B,) values -> (B, 1, 24, 24) images. Bar height 2 (v + 1) pixels, random shift, contrast and noise."""
    b = len(v1)
    rows = torch.arange(SIZE).view(1, SIZE, 1)
    cols = torch.arange(SIZE).view(1, 1, SIZE)
    shift = torch.randint(-1, 2, (b, 1, 1), generator=g)
    img = torch.zeros(b, SIZE, SIZE)
    for v, x0 in ((v1, 4), (v2, 14)):
        top = SIZE - 2 * (v.view(b, 1, 1) + 1)
        inside = (rows >= top) & (cols >= x0 + shift) & (cols < x0 + 6 + shift)
        img = img + inside.float()
    contrast = 0.6 + 0.4 * torch.rand(b, 1, 1, generator=g)
    img = img * contrast + noise * torch.randn(b, SIZE, SIZE, generator=g)
    return img[:, None]


@dataclass
class Batch:
    images: torch.Tensor     # (B, 1, 24, 24)
    q: torch.Tensor          # (B,) SUM or DIFF
    v1: torch.Tensor         # (B,)
    v2: torch.Tensor
    ans: torch.Tensor

    def blank(self) -> "Batch":
        return Batch(torch.zeros_like(self.images), self.q, self.v1, self.v2, self.ans)


def make_batch(n: int, g: torch.Generator) -> Batch:
    v1 = torch.randint(0, N_VALUES, (n,), generator=g)
    v2 = torch.randint(0, N_VALUES, (n,), generator=g)
    q = torch.where(torch.rand(n, generator=g) < 0.5, SUM, DIFF)
    return Batch(render(v1, v2, g), q, v1, v2, answer_of(q, v1, v2))


# ---------- model ----------

class ChartVLM(nn.Module):
    def __init__(self, d: int = 64, layers: int = 2):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(1, 16, 3, 2, 1), nn.ReLU(), nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU())
        self.connector = nn.Linear(32, d)
        self.embed = nn.Embedding(VOCAB, d)
        self.pos = nn.Parameter(torch.randn(N_VIS + 5, d) * 0.02)
        block = nn.TransformerEncoderLayer(d, 4, 2 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        self.head = nn.Linear(d, VOCAB)

    def part_of(self, name: str) -> str:
        return "encoder" if name.startswith("encoder") else "connector" if name.startswith("connector") else "llm"

    def visual_features(self, images: torch.Tensor) -> torch.Tensor:
        """E(I): (B, 1, 24, 24) -> (B, 36, 32)."""
        return self.encoder(images).flatten(2).transpose(1, 2)

    def forward(self, images: torch.Tensor, q: torch.Tensor, prefix: torch.Tensor) -> torch.Tensor:
        """Logits for the answer tokens after [vis, q, BOS, prefix]: (B, 1 + prefix_len, N_NUM)."""
        vis = self.connector(self.visual_features(images))                     # v = P(E(I))
        text = self.embed(torch.cat([q[:, None], torch.full_like(q[:, None], BOS), prefix], 1))
        x = torch.cat([vis, text], 1)
        x = x + self.pos[: x.shape[1]]
        n = x.shape[1]
        causal = torch.triu(torch.full((n, n), float("-inf")), 1)
        h = self.blocks(x, mask=causal)
        return self.head(h[:, N_VIS + 1:])[..., :N_NUM]                        # from the BOS position on

    def token_logprobs(self, b: Batch, tokens: torch.Tensor) -> torch.Tensor:
        """log pi of each of the 3 given tokens: (B, 3)."""
        logits = self(b.images, b.q, tokens[:, :2])
        return logits.log_softmax(-1).gather(-1, tokens[:, :, None]).squeeze(-1)

    @torch.no_grad()
    def generate(self, b: Batch, sample: bool, g: torch.Generator | None = None,
                 force_reads: bool = False) -> torch.Tensor:
        """(B, 3) tokens. force_reads=True writes the true readings, as an oracle perception would."""
        tokens = torch.zeros(len(b.q), 0, dtype=torch.long)
        for t in range(3):
            if force_reads and t < 2:
                nxt = (b.v1, b.v2)[t]
            else:
                logits = self(b.images, b.q, tokens)[:, -1]
                nxt = torch.multinomial(logits.softmax(-1), 1, generator=g)[:, 0] if sample else logits.argmax(-1)
            tokens = torch.cat([tokens, nxt[:, None]], 1)
        return tokens


def set_trainable(model: ChartVLM, mode: str) -> None:
    """mode: 'llm' (E and P frozen), 'connector+llm' (E frozen) or 'full'."""
    open_parts = {"llm": {"llm"}, "connector+llm": {"connector", "llm"}, "full": set(PARTS)}[mode]
    for name, p in model.named_parameters():
        p.requires_grad_(model.part_of(name) in open_parts)


def grad_norms(model: ChartVLM) -> dict[str, float]:
    sq = {p: 0.0 for p in PARTS}
    for name, p in model.named_parameters():
        if p.grad is not None:
            sq[model.part_of(name)] += float(p.grad.pow(2).sum())
    return {k: v ** 0.5 for k, v in sq.items()}


# ---------- rewards ----------

def step_correct(b: Batch, tokens: torch.Tensor) -> torch.Tensor:
    """(B, 3) in {0, 1}: read1 right, read2 right, answer right GIVEN the model's own readings (a calculator check)."""
    r1, r2, a = tokens.unbind(-1)
    return torch.stack([r1 == b.v1, r2 == b.v2, a == answer_of(b.q, r1, r2)], -1).float()


def outcome(b: Batch, tokens: torch.Tensor) -> torch.Tensor:
    return (tokens[:, 2] == b.ans).float()


# ---------- training ----------

def sft(model: ChartVLM, steps: int, n_examples: int, g: torch.Generator, lr: float = 3e-3, bs: int = 64) -> None:
    """Supervised warm start on a small fixed set with the full chain (read1, read2, answer) as targets."""
    data = make_batch(n_examples, g)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        i = torch.randint(0, n_examples, (bs,), generator=g)
        b = Batch(data.images[i], data.q[i], data.v1[i], data.v2[i], data.ans[i])
        target = torch.stack([b.v1, b.v2, b.ans], 1)
        loss = -model.token_logprobs(b, target).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()


@torch.no_grad()
def evaluate(model: ChartVLM, b: Batch) -> dict[str, float]:
    """The post's four controls, greedy decoding, on one held-out batch."""
    model.eval()
    real = model.generate(b, sample=False)
    out = {
        "A_img": float(outcome(b, real).mean()),
        "A_blank": float(outcome(b, model.generate(b.blank(), sample=False)).mean()),
        "A_oracle": float(outcome(b, model.generate(b, sample=False, force_reads=True)).mean()),
        "P": float(step_correct(b, real)[:, :2].all(-1).float().mean()),
    }
    model.train()
    return out


@dataclass
class RLConfig:
    reward: str = "outcome"          # 'outcome' or 'step'
    trainable: str = "connector+llm"
    dynamic: bool = False            # DAPO-style: drop zero-advantage groups, refill from extra prompts
    steps: int = 200
    prompts: int = 32
    k: int = 8
    lr: float = 1e-3


def grpo_step(model: ChartVLM, opt, cfg: RLConfig, g: torch.Generator) -> dict[str, float]:
    """Sample K rollouts per image, score them, normalise inside each group, take one policy-gradient step."""
    n = cfg.prompts * (3 if cfg.dynamic else 1)
    b = make_batch(n, g)
    rep = lambda t: t.repeat_interleave(cfg.k, 0)                               # noqa: E731
    rb = Batch(rep(b.images), rep(b.q), rep(b.v1), rep(b.v2), rep(b.ans))
    tokens = model.generate(rb, sample=True, g=g)                               # (n K, 3)
    steps = step_correct(rb, tokens).view(n, cfg.k, 3)
    win = outcome(rb, tokens).view(n, cfg.k)
    if cfg.reward == "outcome":
        adv = group_advantages(win)[..., None].expand(n, cfg.k, 3)              # same advantage on every token
        flat = win
    else:
        adv = group_advantages(steps.transpose(1, 2)).transpose(1, 2)           # each step normalised on its own
        flat = steps.mean(-1)
    zero = zero_advantage(flat)
    keep = torch.arange(n)
    if cfg.dynamic:
        keep = keep[~zero][: cfg.prompts]
        if len(keep) == 0:
            keep = torch.arange(cfg.prompts)
    rows = (keep[:, None] * cfg.k + torch.arange(cfg.k)).flatten()
    sub = Batch(rb.images[rows], rb.q[rows], rb.v1[rows], rb.v2[rows], rb.ans[rows])
    logp = model.token_logprobs(sub, tokens[rows])                              # (len(rows), 3)
    loss = -(adv[keep].reshape(-1, 3) * logp).sum(-1).mean()
    opt.zero_grad()
    loss.backward()
    norms = grad_norms(model)
    opt.step()
    lucky = (win > 0) & (steps[..., :2].min(-1).values < 1)                     # right answer, wrong reading
    return {"reward": float(win.mean()), "useful": float((~zero[: cfg.prompts]).float().mean()),
            "lucky": float(lucky.sum() / win.sum().clamp_min(1)), "rollouts": n * cfg.k, **norms}


def train_rl(model: ChartVLM, cfg: RLConfig, g: torch.Generator, eval_batch: Batch, eval_every: int = 25):
    """Returns per-step logs and evaluations every `eval_every` steps."""
    set_trainable(model, cfg.trainable)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=cfg.lr)
    logs, evals = [], [{"step": 0, **evaluate(model, eval_batch)}]
    for step in range(1, cfg.steps + 1):
        logs.append(grpo_step(model, opt, cfg, g))
        if step % eval_every == 0 or step == cfg.steps:
            evals.append({"step": step, **evaluate(model, eval_batch)})
    return logs, evals


def encoder_probe(model: ChartVLM, g: torch.Generator, n: int = 2000, steps: int = 300) -> float:
    """Linear-probe accuracy for (v1, v2) on frozen, flattened encoder features E(I)."""
    with torch.no_grad():
        tr, te = make_batch(n, g), make_batch(1000, g)
        f_tr, f_te = model.visual_features(tr.images).flatten(1), model.visual_features(te.images).flatten(1)
        mu, sd = f_tr.mean(0), f_tr.std(0) + 1e-5
        f_tr, f_te = (f_tr - mu) / sd, (f_te - mu) / sd
    probe = nn.Linear(f_tr.shape[1], 2 * N_VALUES)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-2, weight_decay=1e-3)
    for _ in range(steps):
        out = probe(f_tr).view(n, 2, N_VALUES)
        loss = F.cross_entropy(out.reshape(-1, N_VALUES), torch.stack([tr.v1, tr.v2], 1).flatten())
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = probe(f_te).view(-1, 2, N_VALUES).argmax(-1)
        return float((pred == torch.stack([te.v1, te.v2], 1)).float().mean())
