"""Linear probes, ridge reconstruction and k-nearest-neighbour overlap, in plain torch."""
from __future__ import annotations

import torch


def standardize(train: torch.Tensor, test: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mu, sd = train.mean(0), train.std(0, unbiased=False) + 1e-6
    return (train - mu) / sd, (test - mu) / sd


def logistic_probe(x_tr, y_tr, x_te, y_te, n_classes: int, l2: float = 1e-2, steps: int = 300) -> float:
    """Multinomial logistic regression trained with L-BFGS on frozen features. Returns test accuracy."""
    x_tr, x_te = standardize(x_tr.float(), x_te.float())
    torch.manual_seed(0)
    w = torch.zeros(x_tr.shape[1], n_classes, requires_grad=True)
    b = torch.zeros(n_classes, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=0.5, max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(x_tr @ w + b, y_tr) + l2 * (w ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        return float(((x_te @ w + b).argmax(-1) == y_te).float().mean())


def ridge(x: torch.Tensor, y: torch.Tensor, lam: float = 1e-3) -> torch.Tensor:
    """W minimising |XW - Y|^2 + lam * n * |W|^2, with a bias column appended to X."""
    x = torch.cat([x, torch.ones(len(x), 1, dtype=x.dtype)], 1).double()
    a = x.T @ x + lam * len(x) * torch.eye(x.shape[1], dtype=x.dtype)
    return torch.linalg.solve(a, x.T @ y.double()).float()


def ridge_predict(w: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.cat([x, torch.ones(len(x), 1, dtype=x.dtype)], 1) @ w


def knn_overlap(a: torch.Tensor, b: torch.Tensor, k: int = 10) -> torch.Tensor:
    """Per row: the fraction of its k nearest neighbours (cosine) in space a that are also among them in b."""
    def knn(z):
        z = torch.nn.functional.normalize(z.float(), dim=-1)
        s = z @ z.T
        s.fill_diagonal_(-2)
        return s.topk(k, dim=-1).indices
    na, nb = knn(a), knn(b)
    return torch.tensor([len(set(na[i].tolist()) & set(nb[i].tolist())) / k for i in range(len(a))])
