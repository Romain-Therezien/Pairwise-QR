"""Evaluation metrics.

- ``d2_score``: relative improvement in pinball loss over the constant
  unconditional-quantile baseline (Eq. (6), the quantile analogue of R^2).
  The baseline quantile is taken from the *training* targets, so the constant
  predictor gets no oracle knowledge of the evaluation set.
- ``empirical_coverage``: fraction of targets at or below the prediction,
  which should be approximately tau for a well-calibrated model. Compute it
  on held-out data.
"""

from __future__ import annotations

import numpy as np

from .losses import pinball_loss_np


def d2_score(
    y_eval: np.ndarray,
    y_pred: np.ndarray,
    tau: float,
    y_train: np.ndarray,
) -> float:
    """D^2_tau = 1 - L(model) / L(constant train-quantile predictor)."""
    baseline = np.quantile(y_train, tau)
    loss_model = pinball_loss_np(y_eval, y_pred, tau)
    loss_const = pinball_loss_np(y_eval, np.full_like(y_eval, baseline), tau)
    return 1.0 - loss_model / loss_const


def empirical_coverage(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of observations with y <= q_hat (target: tau)."""
    return float(np.mean(np.asarray(y_true) <= np.asarray(y_pred)))


def mae_vs_truth(true_q: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error against the (MC-estimated) true quantile, Eq. (25)."""
    return float(np.mean(np.abs(np.asarray(true_q) - np.asarray(y_pred))))
