"""Quantile regression models.

One model is trained per quantile level tau (feature effects vary across
quantiles; see Section 4.2 of the paper). All constructors accept a seed so
results are exactly reproducible, including torch weight initialization and
DataLoader shuffling.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from .losses import pinball_loss_torch


class NNQuantileRegressor(nn.Module):
    """Feedforward quantile regressor trained with the pinball loss."""

    def __init__(self, input_dim: int, hidden_units=(32, 32), tau: float = 0.5):
        super().__init__()
        self.tau = tau
        layers, prev = [], input_dim
        for h in hidden_units:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).squeeze(-1)

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        self.eval()
        return self(torch.as_tensor(X, dtype=torch.float32)).numpy()


def train_nn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    tau: float,
    hidden_units=(32, 32),
    n_epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-2,
    seed: int = 42,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    log_every: int = 0,
) -> tuple[NNQuantileRegressor, dict]:
    """Train one NN quantile regressor at level tau.

    Returns the model and a history dict with per-epoch train (and, if a
    validation set is given, validation) pinball losses.
    """
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)

    model = NNQuantileRegressor(X_train.shape[1], hidden_units, tau)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    X_t = torch.as_tensor(X_train, dtype=torch.float32)
    y_t = torch.as_tensor(y_train, dtype=torch.float32)
    loader = DataLoader(
        TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True, generator=g
    )

    history = {"train": [], "val": []}
    for epoch in range(n_epochs):
        model.train()
        for X_b, y_b in loader:
            optimizer.zero_grad()
            loss = pinball_loss_torch(y_b, model(X_b), tau)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            history["train"].append(pinball_loss_torch(y_t, model(X_t), tau).item())
            if X_val is not None:
                Xv = torch.as_tensor(X_val, dtype=torch.float32)
                yv = torch.as_tensor(y_val, dtype=torch.float32)
                history["val"].append(pinball_loss_torch(yv, model(Xv), tau).item())
        if log_every and (epoch + 1) % log_every == 0:
            msg = f"[tau={tau:.2f}] epoch {epoch + 1}/{n_epochs} train {history['train'][-1]:.4f}"
            if history["val"]:
                msg += f" val {history['val'][-1]:.4f}"
            print(msg)
    return model, history


def make_gbm(tau: float, seed: int = 42, **overrides):
    """GradientBoostingRegressor with the quantile loss.

    Defaults are the best parameters from the grid search of Appendix C.
    """
    from sklearn.ensemble import GradientBoostingRegressor

    params = dict(
        loss="quantile",
        alpha=tau,
        learning_rate=0.01,
        max_depth=5,
        min_samples_leaf=5,
        n_estimators=600,
        subsample=0.7,
        random_state=seed,
    )
    params.update(overrides)
    return GradientBoostingRegressor(**params)


def make_lightgbm(tau: float, seed: int = 42, **overrides):
    """LGBMRegressor with the quantile objective.

    Defaults are the best parameters from the grid search of Appendix C.
    Note: uses ``verbosity=-1`` (the ``fit(verbose=...)`` kwarg was removed
    in LightGBM 4.x).
    """
    from lightgbm import LGBMRegressor

    params = dict(
        objective="quantile",
        alpha=tau,
        learning_rate=0.05,
        max_depth=6,
        min_child_samples=20,
        num_leaves=63,
        reg_lambda=0,
        subsample=0.7,
        random_state=seed,
        verbosity=-1,
    )
    params.update(overrides)
    return LGBMRegressor(**params)
