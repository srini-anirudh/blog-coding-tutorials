"""Render text lines at a chosen size, and read them back with a small CRNN trained with CTC loss.

Sections "The pipeline tradition" (CRNN + CTC) and "Resolution and the token budget" (small text dissolves).
The font is Pillow's bundled default (Aileron), so the renders are the same on any machine.
"""

from __future__ import annotations

import random
import string

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont

CHARSET = string.digits + string.ascii_lowercase + string.ascii_uppercase + ".,-"
BLANK = 0  # CTC blank; character i of CHARSET has label i + 1
HEIGHT, WIDTH = 32, 224


def random_text(rng: random.Random) -> str:
    """An amount like 1,580.00, a word, or a code like INV-2041: the strings a page reader must get exactly right."""
    kind = rng.random()
    if kind < 0.4:
        whole = rng.randint(0, 99999)
        return f"{whole:,}.{rng.randint(0, 99):02d}"
    if kind < 0.8:
        word = "".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(3, 9)))
        return word.capitalize() if rng.random() < 0.3 else word
    return "".join(rng.choice(string.ascii_uppercase) for _ in range(3)) + "-" + str(rng.randint(100, 9999))


def render_line(text: str, px_per_em: float, oversample: int = 1) -> Image.Image:
    """Draw black text on white with an em of `px_per_em` pixels, as a rasterizer at that resolution would.
    With oversample > 1 the line is drawn that many times larger and then shrunk, which is what downscaling a
    higher-resolution page does."""
    size = px_per_em * oversample
    font = ImageFont.load_default(size=size)
    x0, y0, x1, y1 = font.getbbox(text)
    pad = max(1, round(0.15 * size))
    img = Image.new("L", (x1 - x0 + 2 * pad, round(1.25 * size) + 2 * pad), 255)
    ImageDraw.Draw(img).text((pad - x0, pad), text, font=font, fill=0)
    if oversample > 1:
        img = img.resize((max(1, round(img.width / oversample)), max(1, round(img.height / oversample))),
                         Image.Resampling.BOX)
    return img


def downscale(img: Image.Image, factor: float) -> Image.Image:
    """Shrink a rendered line by `factor` (< 1), as when a whole page is resized to fit a small input."""
    return img.resize((max(1, round(img.width * factor)), max(1, round(img.height * factor))), Image.Resampling.BOX)


def to_tensor(img: Image.Image) -> torch.Tensor:
    """Scale the line to the recognizer's height (enlarging never adds detail), then pad or crop to a fixed width."""
    w = max(1, round(img.width * HEIGHT / img.height))
    img = img.resize((min(w, WIDTH), HEIGHT), Image.Resampling.BILINEAR)
    canvas = Image.new("L", (WIDTH, HEIGHT), 255)
    canvas.paste(img, (0, 0))
    return torch.from_numpy(1.0 - np.asarray(canvas, dtype=np.float32) / 255.0).unsqueeze(0)


def encode(text: str) -> list[int]:
    return [CHARSET.index(c) + 1 for c in text]


def greedy_decode(logits: torch.Tensor) -> list[str]:
    """CTC greedy decoding: best label per frame, merge repeats, drop blanks. logits: (T, B, C)."""
    out = []
    for seq in logits.argmax(-1).T.tolist():
        chars, prev = [], BLANK
        for k in seq:
            if k != prev and k != BLANK:
                chars.append(CHARSET[k - 1])
            prev = k
        out.append("".join(chars))
    return out


class CRNN(nn.Module):
    """Convolutions read the line image, a bidirectional LSTM models the sequence, a linear layer scores characters
    per column. CTC loss aligns the column scores with the text without per-character positions (Shi et al., 2015)."""

    def __init__(self, n_classes: int = len(CHARSET) + 1, hidden: int = 128):
        super().__init__()

        def block(cin, cout, pool):
            return [nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(), nn.MaxPool2d(pool)]

        self.conv = nn.Sequential(*block(1, 32, 2), *block(32, 64, 2), *block(64, 128, (2, 1)), *block(128, 128, (2, 1)))
        self.rnn = nn.LSTM(128 * 2, hidden, bidirectional=True, batch_first=False)
        self.head = nn.Linear(2 * hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.conv(x)                           # (B, 128, 2, W/4)
        f = f.flatten(1, 2).permute(2, 0, 1)       # (T, B, 256)
        return self.head(self.rnn(f)[0])           # (T, B, n_classes)


def make_batch(rng: random.Random, n: int, sizes: tuple[float, float]):
    """n random lines, each at a random em size in `sizes`, half drawn directly and half drawn larger and shrunk."""
    texts, imgs = [], []
    for _ in range(n):
        t = random_text(rng)
        em = rng.uniform(*sizes)
        imgs.append(to_tensor(render_line(t, em, oversample=rng.choice((1, 3)))))
        texts.append(t)
    targets = torch.tensor([k for t in texts for k in encode(t)])
    lengths = torch.tensor([len(t) for t in texts])
    return torch.stack(imgs), targets, lengths, texts


def train_crnn(steps: int = 3000, batch: int = 64, sizes=(3.0, 28.0), seed: int = 0, lr: float = 2e-3, log=None):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = CRNN()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    ctc = nn.CTCLoss(blank=BLANK, zero_infinity=True)
    model.train()
    for step in range(1, steps + 1):
        x, targets, lengths, _ = make_batch(rng, batch, sizes)
        logits = model(x)
        loss = ctc(logits.log_softmax(-1), targets, torch.full((batch,), logits.shape[0]), lengths)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()
        if log and (step % 250 == 0 or step == 1):
            log(f"step {step:5d}  ctc loss {loss.item():.3f}")
    model.eval()
    return model


@torch.no_grad()
def read_lines(model: CRNN, images: list[Image.Image]) -> list[str]:
    return greedy_decode(model(torch.stack([to_tensor(im) for im in images])))
