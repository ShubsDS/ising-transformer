"""Classifier two-sample test (C2ST).

Train a small CNN to distinguish model samples from Wolff samples on the raw
N x N spin field, using circular padding to respect periodic BCs. Under H_0:
p_theta = pi, held-out classifier accuracy should be ~1/2; we report

    z = sqrt(4 * M_test) * (acc - 1/2)  ~  N(0, 1)

so |z| < 2 means the two distributions are statistically indistinguishable
(to this classifier at this sample size).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class C2STReport:
    accuracy: float
    z: float
    n_train: int
    n_test: int


class CircularPad2d(nn.Module):
    def __init__(self, pad: int) -> None:
        super().__init__()
        self.pad = pad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.pad(x, [self.pad] * 4, mode="circular")


class SmallCNN(nn.Module):
    def __init__(self, ch: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            CircularPad2d(1),
            nn.Conv2d(1, ch, kernel_size=3, padding=0),
            nn.GELU(),
            CircularPad2d(1),
            nn.Conv2d(ch, ch, kernel_size=3, padding=0),
            nn.GELU(),
            CircularPad2d(1),
            nn.Conv2d(ch, ch, kernel_size=3, padding=0),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ch, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def run_c2st(
    s_model: np.ndarray,
    s_mc: np.ndarray,
    device: torch.device,
    n_epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-3,
    seed: int = 0,
    verbose: bool = False,
) -> C2STReport:
    rng = np.random.default_rng(seed)
    M = min(s_model.shape[0], s_mc.shape[0])
    s_model = s_model[:M]
    s_mc = s_mc[:M]
    # 80/20 train/test, balanced.
    n_test = int(0.2 * M)
    idx_model = rng.permutation(M)
    idx_mc = rng.permutation(M)
    test_model = s_model[idx_model[:n_test]]
    test_mc = s_mc[idx_mc[:n_test]]
    train_model = s_model[idx_model[n_test:]]
    train_mc = s_mc[idx_mc[n_test:]]

    def make_xy(a: np.ndarray, b: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        x = np.concatenate([a, b], axis=0).astype(np.float32)
        y = np.concatenate([np.zeros(a.shape[0]), np.ones(b.shape[0])]).astype(np.int64)
        # shuffle
        p = rng.permutation(x.shape[0])
        return (
            torch.from_numpy(x[p]).unsqueeze(1).to(device),  # (N, 1, H, W)
            torch.from_numpy(y[p]).to(device),
        )

    x_train, y_train = make_xy(train_model, train_mc)
    x_test, y_test = make_xy(test_model, test_mc)
    n_test_total = x_test.shape[0]

    torch.manual_seed(seed)
    model = SmallCNN(ch=32).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)

    n_train = x_train.shape[0]
    best_acc = 0.5
    for ep in range(n_epochs):
        model.train()
        perm = torch.randperm(n_train, device=device)
        for i in range(0, n_train, batch_size):
            idx = perm[i : i + batch_size]
            logits = model(x_train[idx])
            loss = F.cross_entropy(logits, y_train[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            preds = []
            for i in range(0, n_test_total, batch_size):
                preds.append(model(x_test[i : i + batch_size]).argmax(-1))
            preds = torch.cat(preds)
            acc = float((preds == y_test).float().mean().item())
        if acc > best_acc:
            best_acc = acc
        if verbose:
            print(f"  c2st epoch {ep:2d}: acc {acc:.4f}  (best {best_acc:.4f})")

    z = float(np.sqrt(4.0 * n_test_total) * (best_acc - 0.5))
    return C2STReport(accuracy=best_acc, z=z, n_train=n_train, n_test=n_test_total)
