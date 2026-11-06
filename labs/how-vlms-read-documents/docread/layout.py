"""A LayoutLM-style encoder on synthetic receipts: word embedding + 1D position + embeddings of the box corners.

Section "Reading OCR words with their positions" and Figure 3 of the post. The task is to find the amount that
belongs to TOTAL. Every amount is the same token [NUM], so the model cannot pick the largest number; it has to use
where the amount sits relative to the word TOTAL.
"""

from __future__ import annotations

import random

import torch
import torch.nn as nn

ITEMS = ["coffee", "bagel", "tea", "muffin", "soup", "salad", "juice", "water", "toast", "eggs", "rice", "noodles",
         "cake", "milk", "bread", "cookie", "wrap", "fries", "pasta", "pizza"]
KEYS = ["SUBTOTAL", "TAX", "TOTAL", "CASH", "CHANGE", "DISCOUNT", "TIP"]
HEAD = ["STORE", "RECEIPT", "DATE", "TABLE", "SERVER", "THANK", "YOU"]
VOCAB = ["[PAD]", "[UNK]", "[NUM]"] + ITEMS + KEYS + HEAD + ["large", "small", "x2", "x3"]
WORD_ID = {w: i for i, w in enumerate(VOCAB)}
ORDERS = ["rows", "columns", "shuffled"]


def synthetic_receipt(rng: random.Random):
    """Words with boxes on a 0-1000 grid, and the index of the TOTAL amount. Words are in true row order."""
    words: list[tuple[str, list[int]]] = []
    y = rng.randint(20, 60)

    def line(left: list[str], amount: bool):
        nonlocal y
        x = rng.randint(40, 90)
        for w in left:
            width = 18 * len(w) + rng.randint(-5, 5)
            words.append((w, [x, y, x + width, y + 26]))
            x += width + rng.randint(10, 20)
        idx = None
        if amount:
            x1 = rng.randint(880, 960)
            words.append(("[NUM]", [x1 - rng.randint(90, 140), y, x1, y + 26]))
            idx = len(words) - 1
        y += rng.randint(34, 48)
        return idx

    line(rng.sample(HEAD[:2], 2), False)
    line(["DATE"], False)
    for _ in range(rng.randint(2, 6)):
        item = [rng.choice(ITEMS)] + ([rng.choice(["large", "small", "x2", "x3"])] if rng.random() < 0.4 else [])
        line(item, True)
    keys = ["SUBTOTAL", "TAX", "TOTAL"] + rng.sample(["CASH", "CHANGE", "DISCOUNT", "TIP"], rng.randint(1, 3))
    rng.shuffle(keys)
    total = None
    for k in keys:
        i = line([k], True)
        total = i if k == "TOTAL" else total
    line(["THANK", "YOU"], False)
    return words, total


def ocr_order(words, how: str, rng: random.Random) -> list[int]:
    """The order an OCR engine hands the words over. rows: line by line. columns: everything in the left half,
    then everything in the right half (what a column-splitting engine does to a receipt). shuffled: no order."""
    idx = list(range(len(words)))
    if how == "rows":
        return sorted(idx, key=lambda i: (words[i][1][1], words[i][1][0]))
    if how == "columns":
        return sorted(idx, key=lambda i: (words[i][1][0] > 500, words[i][1][1], words[i][1][0]))
    rng.shuffle(idx)
    return idx


def encode_receipt(words, total, order, ocr_typo: bool = False):
    ids, boxes, label = [], [], 0
    for pos, i in enumerate(order):
        w = words[i][0]
        if ocr_typo and w == "TOTAL":
            w = "T0TAL"  # misread by the OCR engine: not in the vocabulary
        ids.append(WORD_ID.get(w, WORD_ID["[UNK]"]))
        boxes.append(words[i][1])
        if i == total:
            label = pos
    return ids, boxes, label


def make_split(n: int, seed: int, orders=ORDERS, ocr_typo: bool = False):
    rng = random.Random(seed)
    out = []
    for k in range(n):
        words, total = synthetic_receipt(rng)
        out.append(encode_receipt(words, total, ocr_order(words, orders[k % len(orders)], rng), ocr_typo))
    return out


def collate(examples):
    n = max(len(e[0]) for e in examples)
    ids = torch.zeros(len(examples), n, dtype=torch.long)
    boxes = torch.zeros(len(examples), n, 4, dtype=torch.long)
    mask = torch.zeros(len(examples), n, dtype=torch.bool)
    for b, (i, bx, _) in enumerate(examples):
        ids[b, :len(i)] = torch.tensor(i)
        boxes[b, :len(i)] = torch.tensor(bx)
        mask[b, :len(i)] = True
    return ids, boxes, mask, torch.tensor([e[2] for e in examples])


class LayoutEncoder(nn.Module):
    """Token embedding = word + 1D position (+ x0, y0, x1, y1 embeddings on the 0-1000 grid when use_2d)."""

    def __init__(self, use_2d: bool, d: int = 64, layers: int = 2, heads: int = 4, max_len: int = 64):
        super().__init__()
        self.use_2d = use_2d
        self.word = nn.Embedding(len(VOCAB), d)
        self.pos = nn.Embedding(max_len, d)
        self.x = nn.Embedding(1001, d)  # shared by x0 and x1, as in LayoutLM
        self.y = nn.Embedding(1001, d)  # shared by y0 and y1
        self.norm = nn.LayerNorm(d)
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.0, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Linear(d, 1)

    def forward(self, ids, boxes, mask):
        h = self.word(ids) + self.pos(torch.arange(ids.shape[1]))[None]
        if self.use_2d:
            h = h + self.x(boxes[..., 0]) + self.y(boxes[..., 1]) + self.x(boxes[..., 2]) + self.y(boxes[..., 3])
        h = self.encoder(self.norm(h), src_key_padding_mask=~mask)
        return self.head(h).squeeze(-1).masked_fill(~mask, -1e9)  # one score per token: "this is the total"


def train_layout(use_2d: bool, train, steps: int = 1500, batch: int = 32, seed: int = 0, lr: float = 1e-3):
    torch.manual_seed(seed)
    model = LayoutEncoder(use_2d)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        pick = torch.randint(0, len(train), (batch,), generator=g).tolist()
        ids, boxes, mask, label = collate([train[i] for i in pick])
        loss = nn.functional.cross_entropy(model(ids, boxes, mask), label)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    return model


@torch.no_grad()
def accuracy(model: LayoutEncoder, examples) -> float:
    ids, boxes, mask, label = collate(examples)
    return (model(ids, boxes, mask).argmax(-1) == label).float().mean().item()
