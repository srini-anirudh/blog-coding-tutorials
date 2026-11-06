"""One 16 x 16 image task, three architectures: a dense MLP, a small CNN and a tiny ViT.

The post's image lab says a shift breaks the MLP, moves the CNN's feature map with the object, and leaves
the ViT to learn translation from data. `shapes_batch` makes that testable: objects appear only in the
left half during training and only in the right half at test time.
"""

from __future__ import annotations

import torch
from torch import nn

SIZE, SHAPE = 16, 5


def _patterns() -> torch.Tensor:
    p = torch.zeros(4, SHAPE, SHAPE)
    p[0, 2, :] = 1
    p[0, :, 2] = 1  # plus
    p[1, 0, :] = p[1, -1, :] = p[1, :, 0] = p[1, :, -1] = 1  # square outline
    p[2] = torch.eye(SHAPE) + torch.eye(SHAPE).flip(1)
    p[2, 2, 2] = 1  # X
    p[3, :, 0] = 1
    p[3, -1, :] = 1  # L
    return p


PATTERNS = _patterns()


def shapes_batch(batch: int, cols: tuple[int, int], generator: torch.Generator | None = None, noise: float = 0.1):
    """Images (B, 1, 16, 16) with one shape whose top-left column is in [cols[0], cols[1]], any row."""
    labels = torch.randint(0, 4, (batch,), generator=generator)
    rows = torch.randint(0, SIZE - SHAPE + 1, (batch,), generator=generator)
    cs = torch.randint(cols[0], cols[1] + 1, (batch,), generator=generator)
    x = noise * torch.randn(batch, 1, SIZE, SIZE, generator=generator)
    for i in range(batch):
        r, c = rows[i].item(), cs[i].item()
        x[i, 0, r:r + SHAPE, c:c + SHAPE] += PATTERNS[labels[i]]
    return x, labels


TRAIN_COLS = (0, 5)  # the object never reaches the right half in training
SHIFTED_COLS = (6, 11)  # and only appears there at test time


class MLP(nn.Module):
    def __init__(self, hidden: int = 128):
        super().__init__()
        self.body = nn.Sequential(nn.Flatten(), nn.Linear(SIZE * SIZE, hidden), nn.ReLU())  # 256 x 128 = 32,768 weights
        self.head = nn.Linear(hidden, 4)

    def features(self, x):
        return self.body(x)

    def forward(self, x):
        return self.head(self.features(x))


class CNN(nn.Module):
    def __init__(self, c: int = 32, padding_mode: str = "zeros"):
        super().__init__()
        conv = lambda i, o: nn.Conv2d(i, o, 3, padding=1, padding_mode=padding_mode)  # noqa: E731
        self.convs = nn.Sequential(conv(1, 16), nn.ReLU(), conv(16, c), nn.ReLU(), conv(c, c), nn.ReLU())
        self.head = nn.Linear(c, 4)

    def feature_map(self, x):
        return self.convs(x)

    def features(self, x):
        return self.feature_map(x).amax(dim=(2, 3))  # global max pool: the step that buys invariance

    def forward(self, x):
        return self.head(self.features(x))


class TinyViT(nn.Module):
    def __init__(self, patch: int = 4, d: int = 32, layers: int = 2):
        super().__init__()
        self.patch = patch
        n = (SIZE // patch) ** 2
        self.embed = nn.Linear(patch * patch, d)
        self.pos = nn.Parameter(0.02 * torch.randn(n, d))  # position enters only here
        layer = nn.TransformerEncoderLayer(d, 2, 2 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Linear(d, 4)

    def features(self, x):
        p = self.patch
        tokens = x.unfold(2, p, p).unfold(3, p, p).reshape(x.shape[0], -1, p * p)  # (B, 16 patches, 16 pixels)
        return self.encoder(self.embed(tokens) + self.pos).mean(1)

    def forward(self, x):
        return self.head(self.features(x))


def receptive_field_width(n_layers: int, k: int = 3) -> int:
    """Measure, by gradient, how many input columns reach the centre output of n stacked k x k convs."""
    torch.manual_seed(0)
    net = nn.Sequential(*[nn.Conv2d(1, 1, k, padding=k // 2, bias=False) for _ in range(n_layers)])
    for m in net:
        nn.init.constant_(m.weight, 1.0)  # all-positive weights: no accidental cancellation
    x = torch.zeros(1, 1, 33, 33, requires_grad=True)
    net(x)[0, 0, 16, 16].backward()
    return int((x.grad[0, 0, 16] != 0).sum())
