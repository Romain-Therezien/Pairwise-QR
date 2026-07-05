"""Pinball (quantile) loss.

rho_tau(u) = u * (tau - 1{u < 0}) = max(tau * u, (tau - 1) * u).
"""

import numpy as np
import torch


def pinball_loss_torch(y_true: torch.Tensor, y_pred: torch.Tensor, tau: float) -> torch.Tensor:
    """Mean pinball loss at level tau (differentiable, for training)."""
    u = y_true - y_pred
    return torch.mean(torch.maximum(tau * u, (tau - 1.0) * u))


def pinball_loss_np(y_true: np.ndarray, y_pred: np.ndarray, tau: float) -> float:
    """Mean pinball loss at level tau (numpy, for evaluation)."""
    u = np.asarray(y_true, dtype=np.float64) - np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.maximum(tau * u, (tau - 1.0) * u)))
